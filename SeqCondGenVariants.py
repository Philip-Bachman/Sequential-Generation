from __future__ import division, print_function

import logging
import theano
import numpy
import cPickle

from theano import tensor
from collections import OrderedDict

from blocks.graph import ComputationGraph
from blocks.filter import VariableFilter
from blocks.bricks.base import application, _Brick, Brick, lazy
from blocks.bricks.recurrent import BaseRecurrent, recurrent
from blocks.initialization import Constant, IsotropicGaussian, Orthogonal
from blocks.bricks import Random, MLP, Linear, Tanh, Softmax, Sigmoid, Initializable
from blocks.bricks import Tanh, Identity, Activation, Feedforward
from blocks.bricks.cost import BinaryCrossEntropy
from blocks.utils import shared_floatx_nans
from blocks.roles import add_role, WEIGHT, BIAS, PARAMETER, AUXILIARY


from DKCode import get_adam_updates
from HelperFuncs import constFX, to_fX
from LogPDFs import log_prob_bernoulli, gaussian_kld, gaussian_ent


############################################################
############################################################
## ATTENTION-BASED PERCEPTION UNDER TIME CONSTRAINTS      ##
##                                                        ##
## This version of model requires 2d attention.           ##
############################################################
############################################################

class SeqCondGen2d(BaseRecurrent, Initializable, Random):
    """
    SeqCondGen2d -- develops beliefs subject to constraints on perception.

    This model sequentially constructs a conditional density estimate by taking
    repeated glimpses at the input x, and constructing a hypothesis about the
    output y. The objective is maximum likelihood for (x,y) pairs drawn from
    some training set.

    The input pairs (x, y) can be either "static" or "sequential". In the
    static case, the same x and y are used at every step of the hypothesis
    construction loop. In the sequential case, x and y can change at each step
    of the loop.

    ***                                                                     ***
    *** This version of the model assumes the use of a 2d attention module. ***
    ***                                                                     ***
    *** The attention module should accept 5d inputs for specifying the     ***
    *** location, scale, etc. of the attention-based reader.                ***
    ***                                                                     ***

    Parameters:
        x_and_y_are_seqs: boolean telling whether the conditioning information
                          and prediction targets are sequential.
        total_steps: total number of steps in sequential estimation process
        init_steps: number of steps prior to first NLL measurement
        exit_rate: probability of exiting following each non "init" step
                   **^^ THIS IS SET TO 0 WHEN USING SEQUENTIAL INPUT ^^**
        nll_weight: weight for the prediction NLL term at each step.
                   **^^ THIS IS IGNORED WHEN USING STATIC INPUT ^^**
        step_type: whether to use "additive" steps or "jump" steps
                   -- jump steps predict directly from the controller LSTM's
                      "hidden" state (a.k.a. its memory cells).
        x_dim: dimension of inputs on which to condition
        y_dim: dimension of outputs to predict
        att_spec_dim: dimension of the specification for each glimpse
        glimpse_dim: dimension of the read-out from eah glimpse
        glimpse_count: number of glimpses to take per step. the input and output
                       dimensions of the various child networks should be set
                       up to provide glimpse_count glimpse locations, and to
                       receive glimpse_count attention read-outs.
        reader_mlp: used for reading from the input
                    -- this is a 2d attention module for which the attention
                       location is specified by 5 inputs, the first two of
                       which we will assume are (x,y) coordinates.
        writer_mlp: used for writing to the output prediction
                    -- for "add" steps, this takes the controller's visible
                       state as input and updates the current "belief" state.
                       for "jump" steps, this converts the controller's memory
                       state into the current "belief" state.
        con_mlp_in: preprocesses input to the "controller" LSTM
        con_rnn: the "controller" LSTM
        con_mlp_out: CondNet for distribution over z_c2o given con_rnn
        rav_mlp_in: preprocesses input to the "variational controller" LSTM
        rav_rnn: the "variational controller" LSTM
        rav_mlp_out: CondNet for distribution over z_c2o given rav_rnn
        obs_mlp_in: preprocesses input to the "observer" LSTM
        obs_rnn: the "observer" LSTM
        obs_mlp_out: CondNet for distribution over z_o2c given obs_rnn
        var_mlp_in: preprocesses input to the "variational observer" LSTM
        var_rnn: the "variational observer" LSTM
        var_mlp_out: CondNet for distribution over z_o2c given var_rnn
    """
    def __init__(self, x_and_y_are_seqs,
                    total_steps, init_steps,
                    exit_rate, nll_weight,
                    step_type, x_dim, y_dim,
                    att_spec_dim, glimpse_dim, glimpse_count,
                    reader_mlp, writer_mlp,
                    con_mlp_in, con_rnn, con_mlp_out,
                    rav_mlp_in, rav_rnn, rav_mlp_out,
                    obs_mlp_in, obs_rnn, obs_mlp_out,
                    var_mlp_in, var_rnn, var_mlp_out,
                    **kwargs):
        super(SeqCondGen2d, self).__init__(**kwargs)
        if not ((step_type == 'add') or (step_type == 'jump')):
            raise ValueError('step_type must be jump or add')
        # record basic structural parameters
        self.x_and_y_are_seqs = x_and_y_are_seqs
        self.total_steps = total_steps
        self.init_steps = init_steps
        self.exit_rate = exit_rate
        self.nll_weight = nll_weight
        self.step_type = step_type
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.att_spec_dim = att_spec_dim
        self.all_spec_dim = glimpse_count * att_spec_dim
        self.glimpse_dim = glimpse_dim
        self.glimpse_count = glimpse_count
        # construct a sequence of scales for measuring NLL. we'll use scales
        # corresponding to some fixed number of guaranteed steps, followed by
        # a constant probability of early stopping.
        self.nll_scales = self._construct_nll_scales()

        # grab handles for shared read/write models
        self.reader_mlp = reader_mlp
        self.writer_mlp = writer_mlp
        # grab handles for sequential stochastic observe and control models
        self.con_mlp_in = con_mlp_in
        self.con_rnn = con_rnn
        self.con_mlp_out = con_mlp_out
        self.obs_mlp_in = obs_mlp_in
        self.obs_rnn = obs_rnn
        self.obs_mlp_out = obs_mlp_out
        self.rav_mlp_in = rav_mlp_in
        self.rav_rnn = rav_rnn
        self.rav_mlp_out = rav_mlp_out
        self.var_mlp_in = var_mlp_in
        self.var_rnn = var_rnn
        self.var_mlp_out = var_mlp_out

        # set up the transform from belief state to observation space
        self.c_to_y = tensor.nnet.sigmoid

        # create a shared variable switch for controlling sampling
        ones_ary = numpy.ones((1,)).astype(theano.config.floatX)
        self.train_switch = theano.shared(value=ones_ary, name='train_switch')
        # create shared variables for controlling KLd terms
        self.lam_kld_q2p = theano.shared(value=ones_ary, name='lam_kld_q2p')
        self.lam_kld_p2q = theano.shared(value=ones_ary, name='lam_kld_p2q')
        self.set_lam_kld(lam_kld_q2p=0.95, lam_kld_p2q=0.05)
        # create shared variables for controlling optimization/updates
        self.lr = theano.shared(value=0.0001*ones_ary, name='lr')
        self.mom_1 = theano.shared(value=0.9*ones_ary, name='mom_1')
        self.mom_2 = theano.shared(value=0.99*ones_ary, name='mom_2')

        # set noise scale for the attention placement
        self.att_noise = 0.1

        # setup a "null pointer" that will point to the computation graph
        # for this model, which can be built by self.build_model_funcs()...
        self.cg = None

        # record the sub-models around which this model is built
        self.children = [self.reader_mlp, self.writer_mlp,
                         self.con_mlp_in, self.con_rnn, self.con_mlp_out,
                         self.rav_mlp_in, self.rav_rnn, self.rav_mlp_out,
                         self.obs_mlp_in, self.obs_rnn, self.obs_mlp_out,
                         self.var_mlp_in, self.var_rnn, self.var_mlp_out]
        return

    def set_sgd_params(self, lr=0.01, mom_1=0.9, mom_2=0.999):
        """
        Set learning rate and momentum parameter for all updates.
        """
        zero_ary = numpy.zeros((1,))
        # set learning rate
        new_lr = zero_ary + lr
        self.lr.set_value(to_fX(new_lr))
        # set momentums (use first and second order "momentum")
        new_mom_1 = zero_ary + mom_1
        self.mom_1.set_value(to_fX(new_mom_1))
        new_mom_2 = zero_ary + mom_2
        self.mom_2.set_value(to_fX(new_mom_2))
        return

    def set_lam_kld(self, lam_kld_q2p=0.0, lam_kld_p2q=1.0):
        """
        Set the relative weight of various KL-divergence terms.

        kld_q2p: KLd between guide reader and primary reader. KL(q||p)
        kld_p2q: KLd between primary reader and guide reader. KL(p||q)
        """
        zero_ary = numpy.zeros((1,))
        new_lam = zero_ary + lam_kld_q2p
        self.lam_kld_q2p.set_value(to_fX(new_lam))
        new_lam = zero_ary + lam_kld_p2q
        self.lam_kld_p2q.set_value(to_fX(new_lam))
        return

    def _construct_nll_scales(self):
        """
        Construct a sequence of scales for weighting NLL measurements.
        """
        nll_scales = shared_floatx_nans((self.total_steps,), name='nll_scales')
        if self.x_and_y_are_seqs:
            # construct NLL scales for each step, when using sequential x/y
            np_scales = self.nll_weight * numpy.ones((self.total_steps,))
            np_scales[0:self.init_steps] = 0.0
        else:
            # construct NLL scales for each step, when using static x/y
            np_scales = numpy.zeros((self.total_steps,))
            prob_to_get_here = 1.0
            for i in range(self.total_steps):
                if ((i+1) > self.init_steps):
                    np_scales[i] = prob_to_get_here * self.exit_rate
                    prob_to_get_here = prob_to_get_here * (1.0 - self.exit_rate)
            # force exit on the last step -- i.e. assign it any missing weight
            sum_of_scales = numpy.sum(np_scales)
            missing_weight = 1.0 - sum_of_scales
            np_scales[-1] = np_scales[-1] + missing_weight
        nll_scales.set_value(np_scales.astype(theano.config.floatX))
        return nll_scales

    def _allocate(self):
        """
        Allocate shared parameters used by this model.
        """
        # get size information for the desired parameters
        c_dim = self.get_dim('c')
        cc_dim = self.get_dim('c_con')
        hc_dim = self.get_dim('h_con')
        co_dim = self.get_dim('c_obs')
        ho_dim = self.get_dim('h_obs')
        cv_dim = self.get_dim('c_var')
        hv_dim = self.get_dim('h_var')
        cr_dim = self.get_dim('c_rav')
        hr_dim = self.get_dim('h_rav')
        # self.c_0 provides initial belief state
        self.c_0 = shared_floatx_nans((1,c_dim), name='c_0')
        add_role(self.c_0, PARAMETER)
        # self.cc_0/self.hc_0 provides initial state of the controller
        self.cc_0 = shared_floatx_nans((1,cc_dim), name='cc_0')
        add_role(self.cc_0, PARAMETER)
        self.hc_0 = shared_floatx_nans((1,hc_dim), name='hc_0')
        add_role(self.hc_0, PARAMETER)
        # self.co_0/self.ho_0 provides initial state of the observer
        self.co_0 = shared_floatx_nans((1,co_dim), name='co_0')
        add_role(self.co_0, PARAMETER)
        self.ho_0 = shared_floatx_nans((1,ho_dim), name='ho_0')
        add_role(self.ho_0, PARAMETER)
        # self.cv_0/self.hv_0 provides initial state of the variational observer
        self.cv_0 = shared_floatx_nans((1,cv_dim), name='cv_0')
        add_role(self.cv_0, PARAMETER)
        self.hv_0 = shared_floatx_nans((1,hv_dim), name='hv_0')
        add_role(self.hv_0, PARAMETER)
        # self.cr_0/self.hr_0 provides initial state of the variational controller
        self.cr_0 = shared_floatx_nans((1,cr_dim), name='cr_0')
        add_role(self.cr_0, PARAMETER)
        self.hr_0 = shared_floatx_nans((1,hr_dim), name='hr_0')
        add_role(self.hr_0, PARAMETER)
        # add the theano shared variables to our parameter lists
        self.params.extend([ self.c_0, \
                             self.cc_0, self.co_0, self.cv_0, self.cr_0, \
                             self.hc_0, self.ho_0, self.hv_0, self.hr_0 ])
        return

    def _initialize(self):
        # initialize all parameters to zeros...
        for p in self.params:
            p_nan = p.get_value(borrow=False)
            p_zeros = numpy.zeros(p_nan.shape)
            p.set_value(p_zeros.astype(theano.config.floatX))
        return

    def get_dim(self, name):
        if name == 'x':
            return self.x_dim
        elif name == 'nll_scale':
            return 1
        elif name in ['u_o2c', 'z_o2c']:
            return self.obs_mlp_out.output_dim
        elif name in ['u_c2o', 'z_c2o', 'all_spec']:
            return self.all_spec_dim
        elif name in ['c_as_y', 'y']:
            return self.y_dim
        elif name == 'c':
            return self.writer_mlp.output_dim
        elif name == 'h_con':
            return self.con_rnn.get_dim('states')
        elif name == 'c_con':
            return self.con_rnn.get_dim('cells')
        elif name == 'h_obs':
            return self.obs_rnn.get_dim('states')
        elif name == 'c_obs':
            return self.obs_rnn.get_dim('cells')
        elif name == 'h_var':
            return self.var_rnn.get_dim('states')
        elif name == 'c_var':
            return self.var_rnn.get_dim('cells')
        elif name == 'h_rav':
            return self.rav_rnn.get_dim('states')
        elif name == 'c_rav':
            return self.rav_rnn.get_dim('cells')
        elif name in ['nll', 'kl_q2p', 'kl_p2q']:
            return 0
        else:
            super(SeqCondGen2d, self).get_dim(name)
        return

    #------------------------------------------------------------------------

    @recurrent(sequences=['x', 'y', 'u_o2c', 'u_c2o', 'nll_scale'], contexts=[],
               states=['c', 'h_con', 'c_con', 'h_rav', 'c_rav', 'h_obs', 'c_obs', 'h_var', 'c_var', 'z_o2c'],
               outputs=['c', 'h_con', 'c_con', 'h_rav', 'c_rav', 'h_obs', 'c_obs', 'h_var', 'c_var', 'z_o2c', 'c_as_y', 'nll', 'kl_q2p', 'kl_p2q', 'att_map', 'read_img'])
    def iterate(self, x, y, u_o2c, u_c2o, nll_scale, c, h_con, c_con, h_rav, c_rav, h_obs, c_obs, h_var, c_var, z_o2c):
        # Get the current prediction for y
        if self.step_type == 'jump':
            # Controller hidden state tracks belief state (for jump steps)
            c = self.writer_mlp.apply(c_con)
        # Convert from belief state to prediction for y
        c_as_y = self.c_to_y(c)

        ######################################
        # Update the guide controller state. #
        ######################################
        #
        # The guide controller receives both deterministic and stochastic input
        # from the primary observer, and also conditions on the NLL gradient.
        # Its inputs from the primary observer are the same as for the primary
        # controller.
        #
        nll_grad_rav = y - c_as_y # condition on NLL gradient information
        i_rav = self.rav_mlp_in.apply(tensor.concatenate( \
                                      [z_o2c, h_obs, nll_grad_rav], axis=1))
        h_rav, c_rav = self.rav_rnn.apply(states=h_rav, cells=c_rav, \
                                          inputs=i_rav, iterate=False)

        ####################################################################
        # Convert guide and primary controller state into attention specs. #
        ####################################################################
        #
        # The primary and guide controllers both provide stochastic policies
        # over the choice of z_c2o, which specifies the location, scale, etc.
        # of the next "glimpse" that will be fed as input to the primary and
        # guide observers.
        #
        # The policies give the means and log-variances of diagonal Gaussian
        # distributions over the attention spec. The means and log-variances
        # are computed by con_mlp_out(h_con) and rav_mlp_out(h_rav).
        #
        p_z_c2o_mean, p_z_c2o_logvar, p_z_c2o = \
                self.con_mlp_out.apply(h_con, u_c2o)
        q_z_c2o_mean, q_z_c2o_logvar, q_z_c2o = \
                self.rav_mlp_out.apply(h_rav, u_c2o)
        # mix samples from p/q based on value of self.train_switch
        z_c2o = (self.train_switch[0] * q_z_c2o) + \
                ((1.0 - self.train_switch[0]) * p_z_c2o)
        # compute KL(q || p) and KL(p || q) for noisy channel:
        #     controller -> observer  (a.k.a. con to obs, c2o)
        kl_q2p_c2o = tensor.sum(gaussian_kld(q_z_c2o_mean, q_z_c2o_logvar, \
                            p_z_c2o_mean, p_z_c2o_logvar), axis=1)
        kl_p2q_c2o = tensor.sum(gaussian_kld(p_z_c2o_mean, p_z_c2o_logvar, \
                            q_z_c2o_mean, q_z_c2o_logvar), axis=1)

        #######################################################
        # Apply the attention-based reader to the input in x. #
        # -- This may be done for one or many glimpses...     #
        #######################################################
        if self.glimpse_count == 1:
            # shortcut for single glimpse setting
            read_out = self.reader_mlp.read(x, x, z_c2o)
            att_map = self.reader_mlp.att_map(z_c2o)
            read_img = self.reader_mlp.write(read_out, z_c2o)
        else:
            assert False, "SeqCondGen2d only supports single glimpse per frame."

        ################################################
        # Update the primary and guide observer LSTMs. #
        ################################################
        #
        # The primary and guide observers both receive deterministic and
        # stochastic inputs from the primary controller. The deterministic
        # inputs are passed through the primary controller's visible state
        # (i.e. h_con), and the stochastic inputs are passed through z_c2o and
        # read_out. z_c2o gives the sampled attention spec that was used, by
        # the attention-based reader module, to read the value in read_out from
        # the current input x. The guide observer also gets to condition on the
        # current prediction NLL gradient.
        #
        # -- update the primary observer state
        i_gen = self.obs_mlp_in.apply( \
                tensor.concatenate([read_out, z_c2o, h_con], axis=1))
        h_obs, c_obs = self.obs_rnn.apply(states=h_obs, cells=c_obs,
                                          inputs=i_gen, iterate=False)
        # -- update the guide observer state
        nll_grad_var = y - c_as_y # condition on NLL gradient information
        i_var = self.var_mlp_in.apply( \
                tensor.concatenate([read_out, z_c2o, h_con, nll_grad_var], axis=1))
        h_var, c_var = self.var_rnn.apply(states=h_var, cells=c_var,
                                          inputs=i_var, iterate=False)

        ###################################################################
        # Convert guide and primary observer state into samples of z_o2c. #
        ###################################################################
        #
        # The primary and guide observers both provide stochastic policies over
        # the choice of z_o2c, which specifies information to communicate to
        # the primary and guide controllers through a noisy channel.
        #
        # The policies give the means and log-variances of diagonal Gaussian
        # distributions over the values to send to the controllers. The means
        # and log-variances are from obs_mlp_out(h_obs) & var_mlp_out(h_var).
        #
        p_z_o2c_mean, p_z_o2c_logvar, p_z_o2c = \
                self.obs_mlp_out.apply(h_obs, u_o2c)
        q_z_o2c_mean, q_z_o2c_logvar, q_z_o2c = \
                self.var_mlp_out.apply(h_var, u_o2c)
        # mix samples from p/q based on value of self.train_switch
        z_o2c = (self.train_switch[0] * q_z_o2c) + \
                ((1.0 - self.train_switch[0]) * p_z_o2c)
        # compute KL(q || p) and KL(p || q) for noisy channel:
        #     observer -> controller  (a.k.a. obs to con, o2c)
        kl_q2p_o2c = tensor.sum(gaussian_kld(q_z_o2c_mean, q_z_o2c_logvar, \
                                p_z_o2c_mean, p_z_o2c_logvar), axis=1)
        kl_p2q_o2c = tensor.sum(gaussian_kld(p_z_o2c_mean, p_z_o2c_logvar, \
                                q_z_o2c_mean, q_z_o2c_logvar), axis=1)

        #############################################
        # Update the primary controller LSTM state. #
        #############################################
        #
        # The primary controller receives both deterministic and stochastic
        # input from the primary observer. Deterministic input is given by the
        # visible state of the primary observer (i.e. h_obs), and stochastic
        # input is given by the sampled values of z_o2c.
        #
        i_con = self.con_mlp_in.apply(tensor.concatenate([z_o2c, h_obs], axis=1))
        h_con, c_con = self.con_rnn.apply(states=h_con, cells=c_con, \
                                          inputs=i_con, iterate=False)

        ###############################################################
        # Update the "workspace" (maintained either in c or in c_con) #
        ###############################################################
        if self.step_type == 'add':
            # Controller visible state dictates belief updates (for add steps)
            c = c + self.writer_mlp.apply(h_con)
        else:
            # Controller hidden state tracks belief state (for jump steps)
            c = self.writer_mlp.apply(c_con)
        # Convert from belief state to prediction for y
        c_as_y = self.c_to_y(c)

        ######################################################################
        # Compute the NLL of the y prediction as of this step. The NLL at    #
        # each step is rescaled by the (non-negative) weight in nll_scale... #
        ######################################################################
        #
        # Currently, nll_scale is constructed in a fairly simple way, which
        # only depends on whether the model expects to be working with sequences
        # or static input/output (x,y) pairs. A curious user could make fancier
        # versions of self.nll_scales that allow, e.g., predicting sequences of
        # ys which occur at a lower frequency than the xs.
        #
        # E.g., predicting a sequence of digits from a fixed SVHN input image
        # could be done by passing k repeated copies of each digit label as the
        # y sequence, and n*k copies of the input image as the x sequence. We
        # assume k is the number of "glimpses" to take when predicting each
        # digit's label and n is the max number of labels to predict. When the
        # ground truth contains fewer than the max number of labels, the model
        # can be trained to output a "null" label for the extra slots. The key
        # here would be to make nll_scale a vector like [0,0,1,0,0,1] if, e.g.,
        # one wanted to output a prediction every k=3 glimpses.
        #
        nll = -nll_scale * tensor.flatten(log_prob_bernoulli(y, c_as_y))
        #
        # Combine KLd terms from stochastic policies over z_o2c and z_c2o.
        #
        # We currently measure KLd in both directions, for fun and profit.
        #
        kl_q2p = kl_q2p_o2c + kl_q2p_c2o
        kl_p2q = kl_p2q_o2c + kl_p2q_c2o
        return c, h_con, c_con, h_rav, c_rav, h_obs, c_obs, h_var, c_var, z_o2c, c_as_y, nll, kl_q2p, kl_p2q, att_map, read_img

    #------------------------------------------------------------------------

    @application(inputs=['x', 'y'],
                 outputs=['cs', 'c_as_ys', 'nlls', 'kl_q2ps', 'kl_p2qs', 'att_maps', 'read_imgs'])
    def process_inputs(self, x, y):
        # get important size and shape information

        z_o2c_dim = self.get_dim('z_o2c')
        cc_dim = self.get_dim('c_con')
        co_dim = self.get_dim('c_obs')
        cv_dim = self.get_dim('c_var')
        cr_dim = self.get_dim('c_rav')
        hc_dim = self.get_dim('h_con')
        ho_dim = self.get_dim('h_obs')
        hv_dim = self.get_dim('h_var')
        hr_dim = self.get_dim('h_rav')
        as_dim = self.get_dim('all_spec')

        if self.x_and_y_are_seqs:
            batch_size = x.shape[1]
        else:
            # if we're using "static" inputs, then we need to expand them out
            # into a proper sequential form
            batch_size = x.shape[0]
            x = x.dimshuffle('x',0,1).repeat(self.total_steps, axis=0)
            y = y.dimshuffle('x',0,1).repeat(self.total_steps, axis=0)

        # get initial states for all model components
        c0 = self.c_0.repeat(batch_size, axis=0)
        cc0 = self.cc_0.repeat(batch_size, axis=0)
        hc0 = self.hc_0.repeat(batch_size, axis=0)
        co0 = self.co_0.repeat(batch_size, axis=0)
        ho0 = self.ho_0.repeat(batch_size, axis=0)
        cv0 = self.cv_0.repeat(batch_size, axis=0)
        hv0 = self.hv_0.repeat(batch_size, axis=0)
        cr0 = self.cr_0.repeat(batch_size, axis=0)
        hr0 = self.hr_0.repeat(batch_size, axis=0)

        # get noise samples for stochastic channel observer -> controller
        u_o2c = self.theano_rng.normal(
                    size=(self.total_steps, batch_size, z_o2c_dim),
                    avg=0., std=1.)
        z0 = 0. * self.theano_rng.normal(
                    size=(batch_size, z_o2c_dim), avg=0., std=1.)

        # get noise samples for stochastic channel controller -> observer
        u_c2o = self.att_noise * self.theano_rng.normal(
                        size=(self.total_steps, batch_size, as_dim),
                        avg=0., std=1.)

        # run the multi-stage guided generative process
        cs, _, _, _, _, _, _, _, _, _, c_as_ys, nlls, kl_q2ps, kl_p2qs, att_maps, read_imgs = \
                self.iterate(x=x, y=y, u_o2c=u_o2c, u_c2o=u_c2o,
                             nll_scale=self.nll_scales,
                             c=c0,
                             h_con=hc0, c_con=cc0,
                             h_rav=hr0, c_rav=cr0,
                             h_obs=ho0, c_obs=co0,
                             h_var=hv0, c_var=cv0,
                             z_o2c=z0)

        # add name tags to the constructed values
        cs.name = "cs"
        c_as_ys.name = "c_as_ys"
        nlls.name = "nlls"
        kl_q2ps.name = "kl_q2ps"
        kl_p2qs.name = "kl_p2qs"
        att_maps.name = "att_maps"
        read_imgs.name = "read_imgs"
        return cs, c_as_ys, nlls, kl_q2ps, kl_p2qs, att_maps, read_imgs

    def build_model_funcs(self):
        """
        Build the symbolic costs and theano functions relevant to this model.
        """
        # some symbolic vars to represent various inputs/outputs
        if self.x_and_y_are_seqs:
            x_sym = tensor.tensor3('x_sym')
            y_sym = tensor.tensor3('y_sym')
        else:
            x_sym = tensor.matrix('x_sym')
            y_sym = tensor.matrix('y_sym')

        # collect estimates of y given x produced by this model
        cs, c_as_ys, nlls, kl_q2ps, kl_p2qs, att_maps, read_imgs = \
                self.process_inputs(x_sym, y_sym)

        # get the expected NLL part of the VFE bound
        self.nll_term = nlls.sum(axis=0).mean()
        self.nll_term.name = "nll_term"

        # get KL(q || p) and KL(p || q) and KL(p || g)
        self.kld_q2p_term = kl_q2ps.sum(axis=0).mean()
        self.kld_q2p_term.name = "kld_q2p_term"
        self.kld_p2q_term = kl_p2qs.sum(axis=0).mean()
        self.kld_p2q_term.name = "kld_p2q_term"

        # get the proper VFE bound on NLL
        self.nll_bound = self.nll_term + self.kld_q2p_term
        self.nll_bound.name = "nll_bound"

        # grab handles for all the optimizable parameters in our cost
        self.cg = ComputationGraph([self.nll_bound])
        self.joint_params = self.get_model_params(ary_type='theano')

        # apply some l2 regularization to the model parameters
        self.reg_term = (1e-5 * sum([tensor.sum(p**2.0) for p in self.joint_params]))
        self.reg_term.name = "reg_term"

        # compute the full cost w.r.t. which we will optimize params
        self.joint_cost = self.nll_term + \
                          (self.lam_kld_q2p[0] * self.kld_q2p_term) + \
                          (self.lam_kld_p2q[0] * self.kld_p2q_term) + \
                          self.reg_term
        self.joint_cost.name = "joint_cost"

        # get the gradient of the joint cost for all optimizable parameters
        print("Computing gradients of joint_cost...")
        self.joint_grads = OrderedDict()
        grad_list = tensor.grad(self.joint_cost, self.joint_params)
        for i, p in enumerate(self.joint_params):
            self.joint_grads[p] = grad_list[i]

        # construct the updates for all trainable parameters
        self.joint_updates = get_adam_updates(params=self.joint_params, \
                grads=self.joint_grads, alpha=self.lr, \
                beta1=self.mom_1, beta2=self.mom_2, \
                mom2_init=1e-3, smoothing=1e-5, max_grad_norm=10.0)

        # collect the outputs to return from this function
        outputs = [self.joint_cost, self.nll_bound, self.nll_term, \
                   self.kld_q2p_term, self.kld_p2q_term, self.reg_term]
        # collect the required inputs
        inputs = [x_sym, y_sym]

        # compile the theano functions for computing stuff, like for real
        print("Compiling model training/update function...")
        self.train_joint = theano.function(inputs=inputs, \
                                           outputs=outputs, \
                                           updates=self.joint_updates)
        print("Compiling NLL bound estimator function...")
        self.compute_nll_bound = theano.function(inputs=inputs, \
                                                 outputs=outputs)
        return

    def build_attention_funcs(self):
        """
        Build functions for visualizing the behavior of this model, assuming
        self.reader_mlp is SimpleAttentionReader2d or SimpleAttentionReader1d.
        """
        # some symbolic vars to represent various inputs/outputs
        if self.x_and_y_are_seqs:
            x_sym = tensor.tensor3('x_sym_att_funcs')
            y_sym = tensor.tensor3('y_sym_att_funcs')
        else:
            x_sym = tensor.matrix('x_sym_att_funcs')
            y_sym = tensor.matrix('y_sym_att_funcs')
        # collect estimates of y given x produced by this model
        cs, c_as_ys, nlls, kl_q2ps, kl_p2qs, att_maps, read_imgs = \
                self.process_inputs(x_sym, y_sym)
        # build the function for computing the attention trajectories
        print("Compiling attention tracker...")
        inputs = [x_sym, y_sym]
        outputs = [c_as_ys, att_maps, read_imgs]
        sample_func = theano.function(inputs=inputs, \
                                      outputs=outputs)
        def switchy_sampler(x, y, sample_source='q'):
            # store value of sample source switch, to restore later
            old_switch = self.train_switch.get_value()
            if sample_source == 'p':
                # take samples from the primary policy
                zeros_ary = numpy.zeros((1,)).astype(theano.config.floatX)
                self.train_switch.set_value(zeros_ary)
            else:
                # take samples from the guide policy
                ones_ary = numpy.ones((1,)).astype(theano.config.floatX)
                self.train_switch.set_value(ones_ary)
            # sample prediction and attention trajectories
            outs = sample_func(x, y)
            # set sample source switch back to previous value
            self.train_switch.set_value(old_switch)
            # grab prediction values
            y_preds = outs[0]
            # aggregate sequential attention maps
            obs_dim = self.x_dim
            seq_len = self.total_steps
            samp_count = outs[1].shape[1]
            att_count = self.glimpse_count
            a_maps = numpy.zeros((outs[1].shape[0], outs[1].shape[1], obs_dim))
            for a_num in range(att_count):
                start_idx = a_num * obs_dim
                end_idx = (a_num + 1) * obs_dim
                for s1 in range(samp_count):
                    for s2 in range(seq_len):
                        a_maps[s2,s1,:] = a_maps[s2,s1,:] + \
                                          outs[1][s2,s1,start_idx:end_idx]
            # aggregate sequential attention read-outs
            r_outs = numpy.zeros((outs[2].shape[0], outs[2].shape[1], obs_dim))
            for a_num in range(att_count):
                start_idx = a_num * obs_dim
                end_idx = (a_num + 1) * obs_dim
                for s1 in range(samp_count):
                    for s2 in range(seq_len):
                        r_outs[s2,s1,:] = r_outs[s2,s1,:] + \
                                          outs[2][s2,s1,start_idx:end_idx]
            result = [y_preds, a_maps, r_outs]
            return result
        self.sample_attention = switchy_sampler

    def get_model_params(self, ary_type='numpy'):
        """
        Get the optimizable parameters in this model. This returns a list
        and, to reload this model's parameters, the list must stay in order.

        This can provide shared variables or numpy arrays.
        """
        if self.cg is None:
            self.build_model_funcs()
        joint_params = VariableFilter(roles=[PARAMETER])(self.cg.variables)
        if ary_type == 'numpy':
            for i, p in enumerate(joint_params):
                joint_params[i] = p.get_value(borrow=False)
        return joint_params

    def set_model_params(self, numpy_param_list):
        """
        Set the optimizable parameters in this model. This requires a list
        and, to reload this model's parameters, the list must be in order.
        """
        if self.cg is None:
            self.build_model_funcs()
        # grab handles for all the optimizable parameters in our cost
        joint_params = VariableFilter(roles=[PARAMETER])(self.cg.variables)
        for i, p in enumerate(joint_params):
            joint_params[i].set_value(to_fX(numpy_param_list[i]))
        return joint_params

    def save_model_params(self, f_name=None):
        """
        Save model parameters to a pickle file, in numpy form.
        """
        numpy_params = self.get_model_params(ary_type='numpy')
        f_handle = file(f_name, 'wb')
        # dump the dict self.params, which just holds "simple" python values
        cPickle.dump(numpy_params, f_handle, protocol=-1)
        f_handle.close()
        return

    def load_model_params(self, f_name=None):
        """
        Load model parameters from a pickle file, in numpy form.
        """
        pickle_file = open(f_name)
        numpy_params = cPickle.load(pickle_file)
        self.set_model_params(numpy_params)
        pickle_file.close()
        return


############################################################
############################################################
## ATTENTION-BASED PERCEPTION UNDER TIME CONSTRAINTS      ##
##                                                        ##
## This version of model requires 2d attention.           ##
##                                                        ##
## This model is dramatically simplified compared to the  ##
## original SeqCondGen. It should compile faster and use  ##
## less memory. Maybe it'll even work better too?         ##
############################################################
############################################################

class SeqCondGen2dS(BaseRecurrent, Initializable, Random):
    """
    SeqCondGen2dS -- develops beliefs subject to constraints on perception.

    This model sequentially constructs a conditional density estimate by taking
    repeated glimpses at the input x, while formulating a hypothesis about the
    output y. The objective is maximum likelihood for (x,y) pairs drawn from
    some training set.

    The input pairs (x, y) can be either "static" or "sequential". In the
    static case, the same x and y are used at every step of the hypothesis
    construction loop. In the sequential case, x and y can change at each step
    of the loop.

    ***                                                                     ***
    *** This version of the model assumes the use of a 2d attention module. ***
    ***                                                                     ***
    *** The attention module should accept 5d inputs for specifying the     ***
    *** location, scale, etc. of the attention-based reader.                ***
    ***                                                                     ***
    *** Multiple glimpses can be taken in each processing step.             ***
    ***                                                                     ***

    Parameters:
        x_and_y_are_seqs: boolean telling whether the conditioning information
                          and prediction targets are sequential.
        total_steps: total number of steps in sequential estimation process
        init_steps: number of steps prior to first NLL measurement
        exit_rate: probability of exiting following each non "init" step
                   **^^ THIS IS SET TO 0 WHEN USING SEQUENTIAL INPUT ^^**
        nll_weight: weight for the prediction NLL term at each step.
                   **^^ THIS IS IGNORED WHEN USING STATIC INPUT ^^**
        step_type: whether to use "additive" steps or "jump" steps
                   -- jump steps predict directly from the controller LSTM's
                      "hidden" state (a.k.a. its memory cells).
        x_dim: dimension of inputs on which to condition
        y_dim: dimension of outputs to predict
        att_spec_dim: dimension of the specification for each glimpse
        glimpse_dim: dimension of the read-out from each glimpse
        glimpse_count: number of glimpses to take per step. the input and output
                       dimensions of the relevant child networks should be set
                       up to provide glimpse_count glimpse locations, and to
                       receive glimpse_count attention read-outs.
        reader_mlp: used for reading from the input
                    -- this is a 2d attention module for which the attention
                       location is specified by 5 inputs, the first two of
                       which we will assume are (x,y) coordinates.
        writer_mlp: used for writing to the output prediction
                    -- for "add" steps, this takes the controller's visible
                       state as input and updates the current "belief" state.
                       for "jump" steps, this converts the controller's memory
                       state into the current "belief" state.
        con_mlp_in: preprocesses input to the "controller" LSTM
        con_rnn: the "controller" LSTM
        con_mlp_out: CondNet for distribution over z given con_rnn
        rav_mlp_in: preprocesses input to the "variational controller" LSTM
        rav_rnn: the "variational controller" LSTM
        rav_mlp_out: CondNet for distribution over z given rav_rnn
        att_mlp_in: CondNet to convert a subset of z into attention specs
    """
    def __init__(self, x_and_y_are_seqs,
                    total_steps, init_steps,
                    exit_rate, nll_weight,
                    step_type, x_dim, y_dim,
                    att_spec_dim, glimpse_dim, glimpse_count,
                    reader_mlp, writer_mlp,
                    con_mlp_in, con_rnn, con_mlp_out,
                    rav_mlp_in, rav_rnn, rav_mlp_out,
                    att_mlp_in,
                    **kwargs):
        super(SeqCondGen2dS, self).__init__(**kwargs)
        if not ((step_type == 'add') or (step_type == 'jump')):
            raise ValueError('step_type must be jump or add')
        # record basic structural parameters
        self.x_and_y_are_seqs = x_and_y_are_seqs
        self.total_steps = total_steps
        self.init_steps = init_steps
        self.exit_rate = exit_rate
        self.nll_weight = nll_weight
        self.step_type = step_type
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.att_spec_dim = att_spec_dim
        self.all_spec_dim = glimpse_count * att_spec_dim
        self.glimpse_dim = glimpse_dim
        self.glimpse_count = glimpse_count
        # construct a sequence of scales for measuring NLL. we'll use scales
        # corresponding to some fixed number of guaranteed steps, followed by
        # a constant probability of early stopping.
        self.nll_scales = self._construct_nll_scales()

        # grab handles for shared read/write models
        self.reader_mlp = reader_mlp
        self.writer_mlp = writer_mlp
        # grab handles for sequential stochastic observe and control models
        self.con_mlp_in = con_mlp_in
        self.con_rnn = con_rnn
        self.con_mlp_out = con_mlp_out
        self.rav_mlp_in = rav_mlp_in
        self.rav_rnn = rav_rnn
        self.rav_mlp_out = rav_mlp_out
        self.att_mlp_in = att_mlp_in

        # set up the transform from belief state to observation space
        self.c_to_y = tensor.nnet.sigmoid

        # create a shared variable switch for controlling sampling
        ones_ary = numpy.ones((1,)).astype(theano.config.floatX)
        self.train_switch = theano.shared(value=ones_ary, name='train_switch')
        # create shared variables for controlling KLd terms
        self.lam_kld_q2p = theano.shared(value=ones_ary, name='lam_kld_q2p')
        self.lam_kld_p2q = theano.shared(value=ones_ary, name='lam_kld_p2q')
        self.set_lam_kld(lam_kld_q2p=0.95, lam_kld_p2q=0.05)
        # create shared variable for controlling attention entropy
        self.lam_ent_att = theano.shared(value=ones_ary, name='lam_ent_att')
        self.set_lam_ent(lam_ent_att=0.05)
        # create shared variables for controlling optimization/updates
        self.lr = theano.shared(value=0.0001*ones_ary, name='lr')
        self.mom_1 = theano.shared(value=0.9*ones_ary, name='mom_1')
        self.mom_2 = theano.shared(value=0.99*ones_ary, name='mom_2')

        # setup a "null pointer" that will point to the computation graph
        # for this model, which can be built by self.build_model_funcs()...
        self.cg = None

        # record the sub-models around which this model is built
        self.children = [self.reader_mlp, self.writer_mlp,
                         self.con_mlp_in, self.con_rnn, self.con_mlp_out,
                         self.rav_mlp_in, self.rav_rnn, self.rav_mlp_out,
                         self.att_mlp_in]
        return

    def set_sgd_params(self, lr=0.01, mom_1=0.9, mom_2=0.999):
        """
        Set learning rate and momentum parameter for all updates.
        """
        zero_ary = numpy.zeros((1,))
        # set learning rate
        new_lr = zero_ary + lr
        self.lr.set_value(to_fX(new_lr))
        # set momentums (use first and second order "momentum")
        new_mom_1 = zero_ary + mom_1
        self.mom_1.set_value(to_fX(new_mom_1))
        new_mom_2 = zero_ary + mom_2
        self.mom_2.set_value(to_fX(new_mom_2))
        return

    def set_lam_kld(self, lam_kld_q2p=0.0, lam_kld_p2q=1.0):
        """
        Set the relative weight of various KL-divergence terms.

        kld_q2p: KLd between guide and primary controller. KL(q || p)
        kld_p2q: KLd between primary and guide controller. KL(p || q)
        """
        zero_ary = numpy.zeros((1,))
        new_lam = zero_ary + lam_kld_q2p
        self.lam_kld_q2p.set_value(to_fX(new_lam))
        new_lam = zero_ary + lam_kld_p2q
        self.lam_kld_p2q.set_value(to_fX(new_lam))
        return

    def set_lam_ent(self, lam_ent_att=0.0):
        """
        Set the relative weight of entropy reward for attention.
        """
        zero_ary = numpy.zeros((1,))
        new_lam = zero_ary + lam_ent_att
        self.lam_ent_att.set_value(to_fX(new_lam))
        return

    def _construct_nll_scales(self):
        """
        Construct a sequence of scales for weighting NLL measurements.
        """
        nll_scales = shared_floatx_nans((self.total_steps,), name='nll_scales')
        if self.x_and_y_are_seqs:
            # construct NLL scales for each step, when using sequential x/y
            np_scales = self.nll_weight * numpy.ones((self.total_steps,))
            np_scales[0:self.init_steps] = 0.0
        else:
            # construct NLL scales for each step, when using static x/y
            np_scales = numpy.zeros((self.total_steps,))
            prob_to_get_here = 1.0
            for i in range(self.total_steps):
                if ((i+1) > self.init_steps):
                    np_scales[i] = prob_to_get_here * self.exit_rate
                    prob_to_get_here = prob_to_get_here * (1.0 - self.exit_rate)
            # force exit on the last step -- i.e. assign it any missing weight
            sum_of_scales = numpy.sum(np_scales)
            missing_weight = 1.0 - sum_of_scales
            np_scales[-1] = np_scales[-1] + missing_weight
        nll_scales.set_value(np_scales.astype(theano.config.floatX))
        return nll_scales

    def _allocate(self):
        """
        Allocate shared parameters used by this model.
        """
        # get size information for the desired parameters
        c_dim = self.get_dim('c')
        cc_dim = self.get_dim('c_con')
        hc_dim = self.get_dim('h_con')
        cr_dim = self.get_dim('c_rav')
        hr_dim = self.get_dim('h_rav')
        # self.c_0 provides initial belief state
        self.c_0 = shared_floatx_nans((1,c_dim), name='c_0')
        add_role(self.c_0, PARAMETER)
        # self.cc_0/self.hc_0 provides initial state of the controller
        self.cc_0 = shared_floatx_nans((1,cc_dim), name='cc_0')
        add_role(self.cc_0, PARAMETER)
        self.hc_0 = shared_floatx_nans((1,hc_dim), name='hc_0')
        add_role(self.hc_0, PARAMETER)
        # self.cr_0/self.hr_0 provides initial state of the variational controller
        self.cr_0 = shared_floatx_nans((1,cr_dim), name='cr_0')
        add_role(self.cr_0, PARAMETER)
        self.hr_0 = shared_floatx_nans((1,hr_dim), name='hr_0')
        add_role(self.hr_0, PARAMETER)
        # add the theano shared variables to our parameter lists
        self.params.extend([ self.c_0, \
                             self.cc_0, self.cr_0, \
                             self.hc_0, self.hr_0 ])
        return

    def _initialize(self):
        # initialize all parameters to zeros...
        for p in self.params:
            p_nan = p.get_value(borrow=False)
            p_zeros = numpy.zeros(p_nan.shape)
            p.set_value(p_zeros.astype(theano.config.floatX))
        return

    def get_dim(self, name):
        if name == 'x':
            return self.x_dim
        elif name == 'nll_scale':
            return 1
        elif name in ['u', 'z']:
            return self.con_mlp_out.output_dim
        elif name in ['z_att']:
            return self.att_mlp_in.input_dim
        elif name in ['u_att']:
            return self.all_spec_dim
        elif name in ['c_as_y', 'y']:
            return self.y_dim
        elif name == 'c':
            return self.writer_mlp.output_dim
        elif name == 'h_con':
            return self.con_rnn.get_dim('states')
        elif name == 'c_con':
            return self.con_rnn.get_dim('cells')
        elif name == 'h_rav':
            return self.rav_rnn.get_dim('states')
        elif name == 'c_rav':
            return self.rav_rnn.get_dim('cells')
        elif name in ['nll', 'kl_q2p', 'kl_p2q']:
            return 0
        else:
            super(SeqCondGen2dS, self).get_dim(name)
        return

    #------------------------------------------------------------------------

    @recurrent(sequences=['x', 'y', 'u', 'u_att', 'nll_scale'], contexts=[],
               states=['c', 'h_con', 'c_con', 'h_rav', 'c_rav'],
               outputs=['c', 'h_con', 'c_con', 'h_rav', 'c_rav', 'c_as_y', 'nll', 'kl_q2p', 'kl_p2q', 'ent_att', 'att_map', 'read_img'])
    def iterate(self, x, y, u, u_att, nll_scale, c, h_con, c_con, h_rav, c_rav):
        # Get the current prediction for y
        if self.step_type == 'jump':
            # Controller hidden state tracks belief state (for jump steps)
            c = self.writer_mlp.apply(c_con)
        # Convert from belief state to prediction for y
        c_as_y = self.c_to_y(c)

        ######################################
        # Update the guide controller state. #
        ######################################
        #
        # Guide controller can condition on any available information.
        #
        nll_grad_rav = y - c_as_y # condition on NLL gradient information
        i_rav = self.rav_mlp_in.apply(tensor.concatenate( \
                                      [h_con, nll_grad_rav], axis=1))
        h_rav, c_rav = self.rav_rnn.apply(states=h_rav, cells=c_rav, \
                                          inputs=i_rav, iterate=False)

        #################################################################
        # Convert guide and primary controller state into samples of z. #
        #################################################################
        #
        # In this model, z contains two subsets: z_att and z_slf. The values in
        # z_slf are fed back into the main controller LSTM as inputs, to provide
        # stochasticity beyond that which stems from uncertainty in the read
        # operation. The values in z_att are fed into self.att_mlp_in, which
        # transforms them into a set of self.glimpse_count specs for applying
        # the attention-based reader module.
        #
        # By representing stochasticity in the attention placement using a set
        # of latent variables, rather than directly specifying wobble in the
        # attention placement, we can form multi-modal distributions over the
        # attention placement at each step. This multi-modality might be useful
        # when the model must divide its attention among multiple objects.
        #
        p_z_mean, p_z_logvar, p_z = \
                self.con_mlp_out.apply(h_con, u)
        q_z_mean, q_z_logvar, q_z = \
                self.rav_mlp_out.apply(h_rav, u)
        # mix samples from p/q based on value of self.train_switch
        z = (self.train_switch[0] * q_z) + \
            ((1.0 - self.train_switch[0]) * p_z)
        # compute KL(q || p) and KL(p || q) for latent var distributions
        kl_q2p = tensor.sum(gaussian_kld(q_z_mean, q_z_logvar, \
                            p_z_mean, p_z_logvar), axis=1)
        kl_p2q = tensor.sum(gaussian_kld(p_z_mean, p_z_logvar, \
                            q_z_mean, q_z_logvar), axis=1)
        ##################################################
        # Split z into attention and self-control parts. #
        ##################################################
        z_att_dim = self.get_dim('z_att')
        z_att = z[:,:z_att_dim]
        z_slf = z[:,z_att_dim:]
        # transform latent variables z_att into one or more attention specs
        as_mean, as_logvar, att_specs = self.att_mlp_in.apply(z_att, u_att)
        ent_att = tensor.sum(gaussian_ent(as_mean, as_logvar), axis=1)

        #######################################################
        # Apply the attention-based reader to the input in x. #
        # -- This may be done for one or many glimpses. --    #
        #######################################################
        if self.glimpse_count == 1:
            # shortcut for single glimpse setting
            read_out = self.reader_mlp.read(x, x, att_specs)
            att_map = self.reader_mlp.att_map(att_specs)
            read_img = self.reader_mlp.write(read_out, att_specs)
        else:
            # compute the outcomes of multiple simultaneous glimpses
            read_outs = []
            att_maps = []
            read_imgs = []
            #att_locs = []
            for g_num in range(self.glimpse_count):
                as_start = g_num * self.att_spec_dim
                as_end = (g_num + 1) * self.att_spec_dim
                as_i = att_specs[:,as_start:as_end]
                ro_i = self.reader_mlp.read(x, x, as_i)
                read_outs.append(ro_i)
                att_maps.append(self.reader_mlp.att_map(as_i))
                read_imgs.append(self.reader_mlp.write(ro_i, as_i))
                #att_locs.append(as_i[:,0:2])
            read_out = tensor.concatenate(read_outs, axis=1)
            att_map = tensor.concatenate(att_maps, axis=1)
            read_img = tensor.concatenate(read_imgs, axis=1)
            #att_loc = tensor.stack(att_locs)

        #############################################
        # Update the primary controller LSTM state. #
        #############################################
        #
        # The primary controller gets deterministic and stochastic inputs from
        # its prior self at each time step. Deterministic input is its visible
        # state from the previous step. The stochastic inputs are z_slf, which
        # provides a "direct" noisy channel from t -> t+1, and read_out and
        # att_specs. read_out and att_specs are the read-outs and placement
        # specs for multiple applications of the attention module.
        #
        # The primary controller's visible state "h_con" sets distributions
        # over z, which gets split into z_slf and z_att. z_att is transformed
        # into one or more attention specs by self.att_mlp_in.
        #
        i_con = self.con_mlp_in.apply(tensor.concatenate([z_slf, read_out, att_specs], axis=1))
        h_con, c_con = self.con_rnn.apply(states=h_con, cells=c_con, \
                                          inputs=i_con, iterate=False)

        ###############################################################
        # Update the "workspace" (maintained either in c or in c_con) #
        ###############################################################
        if self.step_type == 'add':
            # Controller visible state dictates belief updates (for add steps)
            c = c + self.writer_mlp.apply(h_con)
        else:
            # Controller hidden state tracks belief state (for jump steps)
            c = self.writer_mlp.apply(c_con)
        # Convert from belief state to prediction for y
        c_as_y = self.c_to_y(c)

        ########################################################
        # Compute the NLL of the y prediction as of this step. #
        ########################################################
        nll = -nll_scale * tensor.flatten(log_prob_bernoulli(y, c_as_y))
        return c, h_con, c_con, h_rav, c_rav, c_as_y, nll, kl_q2p, kl_p2q, ent_att, att_map, read_img

    #------------------------------------------------------------------------

    @application(inputs=['x', 'y'],
                 outputs=['cs', 'c_as_ys', 'nlls', 'kl_q2ps', 'kl_p2qs', 'ent_atts', 'att_maps', 'read_imgs'])
    def process_inputs(self, x, y):
        # get important size and shape information
        z_dim = self.get_dim('z')
        cc_dim = self.get_dim('c_con')
        cr_dim = self.get_dim('c_rav')
        hc_dim = self.get_dim('h_con')
        hr_dim = self.get_dim('h_rav')
        as_dim = self.all_spec_dim

        if self.x_and_y_are_seqs:
            batch_size = x.shape[1]
        else:
            # if we're using "static" inputs, then we need to expand them out
            # into a proper sequential form
            batch_size = x.shape[0]
            x = x.dimshuffle('x',0,1).repeat(self.total_steps, axis=0)
            y = y.dimshuffle('x',0,1).repeat(self.total_steps, axis=0)

        # get initial states for all model components
        c0 = self.c_0.repeat(batch_size, axis=0)
        cc0 = self.cc_0.repeat(batch_size, axis=0)
        hc0 = self.hc_0.repeat(batch_size, axis=0)
        cr0 = self.cr_0.repeat(batch_size, axis=0)
        hr0 = self.hr_0.repeat(batch_size, axis=0)

        # get noise samples for stochastic variables
        u = self.theano_rng.normal(
                    size=(self.total_steps, batch_size, z_dim),
                    avg=0., std=1.)

        u_att = 0.2 * self.theano_rng.normal(
                    size=(self.total_steps, batch_size, as_dim),
                    avg=0., std=1.)

        # run the multi-stage guided generative process
        cs, _, _, _, _, c_as_ys, nlls, kl_q2ps, kl_p2qs, ent_atts, att_maps, read_imgs = \
                self.iterate(x=x, y=y, u=u, u_att=u_att,
                             nll_scale=self.nll_scales, c=c0,
                             h_con=hc0, c_con=cc0, h_rav=hr0, c_rav=cr0)

        # add name tags to the constructed values
        cs.name = "cs"
        c_as_ys.name = "c_as_ys"
        nlls.name = "nlls"
        kl_q2ps.name = "kl_q2ps"
        kl_p2qs.name = "kl_p2qs"
        ent_atts.name = "ent_atts"
        att_maps.name = "att_maps"
        read_imgs.name = "read_imgs"
        return cs, c_as_ys, nlls, kl_q2ps, kl_p2qs, ent_atts, att_maps, read_imgs

    def build_model_funcs(self):
        """
        Build the symbolic costs and theano functions relevant to this model.
        """
        # some symbolic vars to represent various inputs/outputs
        if self.x_and_y_are_seqs:
            x_sym = tensor.tensor3('x_sym')
            y_sym = tensor.tensor3('y_sym')
        else:
            x_sym = tensor.matrix('x_sym')
            y_sym = tensor.matrix('y_sym')

        # collect estimates of y given x produced by this model
        cs, c_as_ys, nlls, kl_q2ps, kl_p2qs, ent_atts, att_maps, read_imgs = \
                self.process_inputs(x_sym, y_sym)

        # get the expected NLL part of the VFE bound
        self.nll_term = nlls.sum(axis=0).mean()
        self.nll_term.name = "nll_term"

        # get KL(q || p) and KL(p || q)
        self.kld_q2p_term = self.lam_kld_q2p[0] * kl_q2ps.sum(axis=0).mean()
        self.kld_q2p_term.name = "kld_q2p_term"
        self.kld_p2q_term = self.lam_kld_p2q[0] * kl_p2qs.sum(axis=0).mean()
        self.kld_p2q_term.name = "kld_p2q_term"

        # get attention entropy term
        self.ent_att_term = -self.lam_ent_att[0] * ent_atts.sum(axis=0).mean()

        # compute a cost that depends on all trainable model parameters
        partial_cost = self.nll_term + \
                       self.kld_q2p_term + \
                       self.kld_p2q_term + \
                       self.ent_att_term

        # grab handles for all the optimizable parameters in our cost
        self.cg = ComputationGraph([partial_cost])
        self.joint_params = self.get_model_params(ary_type='theano')

        # apply some l2 regularization to the model parameters
        self.reg_term = (1e-5 * sum([tensor.sum(p**2.0) for p in self.joint_params]))
        self.reg_term.name = "reg_term"

        # compute the full cost w.r.t. which we will optimize params
        self.joint_cost = self.nll_term + \
                          self.kld_q2p_term + \
                          self.kld_p2q_term + \
                          self.ent_att_term + \
                          self.reg_term
        self.joint_cost.name = "joint_cost"

        # get the gradient of the joint cost for all optimizable parameters
        print("Computing gradients of joint_cost...")
        self.joint_grads = OrderedDict()
        grad_list = tensor.grad(self.joint_cost, self.joint_params)
        for i, p in enumerate(self.joint_params):
            self.joint_grads[p] = grad_list[i]

        # construct the updates for all trainable parameters
        self.joint_updates = get_adam_updates(params=self.joint_params,
                grads=self.joint_grads, alpha=self.lr,
                beta1=self.mom_1, beta2=self.mom_2,
                mom2_init=1e-3, smoothing=1e-5, max_grad_norm=5.0)

        # collect the outputs to return from this function
        outputs = [self.joint_cost, self.nll_term, \
                   self.kld_q2p_term, self.kld_p2q_term, \
                   self.ent_att_term, self.reg_term]
        # collect the required inputs
        inputs = [x_sym, y_sym]

        # compile the theano functions for computing stuff, like for real
        print("Compiling model training/update function...")
        self.train_joint = theano.function(inputs=inputs, \
                                           outputs=outputs, \
                                           updates=self.joint_updates)
        print("Compiling NLL bound estimator function...")
        self.compute_nll_bound = theano.function(inputs=inputs, \
                                                 outputs=outputs)
        return

    def build_attention_funcs(self):
        """
        Build functions for visualizing the behavior of this model, assuming
        self.reader_mlp is SimpleAttentionReader2d or SimpleAttentionReader1d.
        """
        # some symbolic vars to represent various inputs/outputs
        if self.x_and_y_are_seqs:
            x_sym = tensor.tensor3('x_sym_att_funcs')
            y_sym = tensor.tensor3('y_sym_att_funcs')
        else:
            x_sym = tensor.matrix('x_sym_att_funcs')
            y_sym = tensor.matrix('y_sym_att_funcs')
        # collect estimates of y given x produced by this model
        cs, c_as_ys, nlls, kl_q2ps, kl_p2qs, ent_atts, att_maps, read_imgs = \
                self.process_inputs(x_sym, y_sym)
        # build the function for computing the attention trajectories
        print("Compiling attention tracker...")
        inputs = [x_sym, y_sym]
        outputs = [c_as_ys, att_maps, read_imgs]
        sample_func = theano.function(inputs=inputs, \
                                      outputs=outputs)
        def switchy_sampler(x, y, sample_source='q'):
            # store value of sample source switch, to restore later
            old_switch = self.train_switch.get_value()
            if sample_source == 'p':
                # take samples from the primary policy
                zeros_ary = numpy.zeros((1,)).astype(theano.config.floatX)
                self.train_switch.set_value(zeros_ary)
            else:
                # take samples from the guide policy
                ones_ary = numpy.ones((1,)).astype(theano.config.floatX)
                self.train_switch.set_value(ones_ary)
            # sample prediction and attention trajectories
            outs = sample_func(x, y)
            # set sample source switch back to previous value
            self.train_switch.set_value(old_switch)
            # grab prediction values
            y_preds = outs[0]
            # aggregate sequential attention maps
            obs_dim = self.x_dim
            seq_len = self.total_steps
            samp_count = outs[1].shape[1]
            att_count = self.glimpse_count
            a_maps = numpy.zeros((outs[1].shape[0], outs[1].shape[1], obs_dim))
            for a_num in range(att_count):
                start_idx = a_num * obs_dim
                end_idx = (a_num + 1) * obs_dim
                for s1 in range(samp_count):
                    for s2 in range(seq_len):
                        a_maps[s2,s1,:] = a_maps[s2,s1,:] + \
                                          outs[1][s2,s1,start_idx:end_idx]
            # aggregate sequential attention read-outs
            r_outs = numpy.zeros((outs[2].shape[0], outs[2].shape[1], obs_dim))
            for a_num in range(att_count):
                start_idx = a_num * obs_dim
                end_idx = (a_num + 1) * obs_dim
                for s1 in range(samp_count):
                    for s2 in range(seq_len):
                        r_outs[s2,s1,:] = r_outs[s2,s1,:] + \
                                          outs[2][s2,s1,start_idx:end_idx]
            result = [y_preds, a_maps, r_outs]
            return result
        self.sample_attention = switchy_sampler
        return

    def get_model_params(self, ary_type='numpy'):
        """
        Get the optimizable parameters in this model. This returns a list
        and, to reload this model's parameters, the list must stay in order.

        This can provide shared variables or numpy arrays.
        """
        if self.cg is None:
            self.build_model_funcs()
        joint_params = VariableFilter(roles=[PARAMETER])(self.cg.variables)
        if ary_type == 'numpy':
            for i, p in enumerate(joint_params):
                joint_params[i] = p.get_value(borrow=False)
        return joint_params

    def set_model_params(self, numpy_param_list):
        """
        Set the optimizable parameters in this model. This requires a list
        and, to reload this model's parameters, the list must be in order.
        """
        if self.cg is None:
            self.build_model_funcs()
        # grab handles for all the optimizable parameters in our cost
        joint_params = VariableFilter(roles=[PARAMETER])(self.cg.variables)
        for i, p in enumerate(joint_params):
            joint_params[i].set_value(to_fX(numpy_param_list[i]))
        return joint_params

    def save_model_params(self, f_name=None):
        """
        Save model parameters to a pickle file, in numpy form.
        """
        numpy_params = self.get_model_params(ary_type='numpy')
        f_handle = file(f_name, 'wb')
        # dump the dict self.params, which just holds "simple" python values
        cPickle.dump(numpy_params, f_handle, protocol=-1)
        f_handle.close()
        return

    def load_model_params(self, f_name=None):
        """
        Load model parameters from a pickle file, in numpy form.
        """
        pickle_file = open(f_name)
        numpy_params = cPickle.load(pickle_file)
        self.set_model_params(numpy_params)
        pickle_file.close()
        return

############################################################
############################################################
## ATTENTION-BASED PERCEPTION UNDER TIME CONSTRAINTS      ##
##                                                        ##
## This version of model requires 2d attention.           ##
##                                                        ##
## This model is dramatically simplified compared to the  ##
## original SeqCondGen. It should compile faster and use  ##
## less memory. Maybe it'll even work better too?         ##
##                                                        ##
## This version of the model adds a direct regression     ##
## cost for shaping attention placement.                  ##
############################################################
############################################################

class SeqCondGen2dSL(BaseRecurrent, Initializable, Random):
    """
    SeqCondGen2dSL -- develops beliefs subject to constraints on perception.

    This model sequentially constructs a conditional density estimate by taking
    repeated glimpses at the input x, while formulating a hypothesis about the
    output y. The objective is maximum likelihood for (x,y) pairs drawn from
    some training set.

    The input pairs (x, y) can be either "static" or "sequential". In the
    static case, the same x and y are used at every step of the hypothesis
    construction loop. In the sequential case, x and y can change at each step
    of the loop.

    ***                                                                     ***
    *** This version of the model assumes the use of a 2d attention module. ***
    ***                                                                     ***
    *** The attention module should accept 5d inputs for specifying the     ***
    *** location, scale, etc. of the attention-based reader.                ***
    ***                                                                     ***
    *** Multiple glimpses can be taken in each processing step.             ***
    ***                                                                     ***

    Parameters:
        x_and_y_are_seqs: boolean telling whether the conditioning information
                          and prediction targets are sequential.
        total_steps: total number of steps in sequential estimation process
        init_steps: number of steps prior to first NLL measurement
        exit_rate: probability of exiting following each non "init" step
                   **^^ THIS IS SET TO 0 WHEN USING SEQUENTIAL INPUT ^^**
        nll_weight: weight for the prediction NLL term at each step.
                   **^^ THIS IS IGNORED WHEN USING STATIC INPUT ^^**
        step_type: whether to use "additive" steps or "jump" steps
                   -- jump steps predict directly from the controller LSTM's
                      "hidden" state (a.k.a. its memory cells).
        x_dim: dimension of inputs on which to condition
        y_dim: dimension of outputs to predict
        att_spec_dim: dimension of the specification for each glimpse
        glimpse_dim: dimension of the read-out from each glimpse
        glimpse_count: number of glimpses to take per step. the input and output
                       dimensions of the relevant child networks should be set
                       up to provide glimpse_count glimpse locations, and to
                       receive glimpse_count attention read-outs.
        reader_mlp: used for reading from the input
                    -- this is a 2d attention module for which the attention
                       location is specified by 5 inputs, the first two of
                       which we will assume are (x,y) coordinates.
        writer_mlp: used for writing to the output prediction
                    -- for "add" steps, this takes the controller's visible
                       state as input and updates the current "belief" state.
                       for "jump" steps, this converts the controller's memory
                       state into the current "belief" state.
        con_mlp_in: preprocesses input to the "controller" LSTM
        con_rnn: the "controller" LSTM
        con_mlp_out: CondNet for distribution over z given con_rnn
        rav_mlp_in: preprocesses input to the "variational controller" LSTM
        rav_rnn: the "variational controller" LSTM
        rav_mlp_out: CondNet for distribution over z given rav_rnn
        att_mlp_in: converts a subset of z into attention specs
    """
    def __init__(self, x_and_y_are_seqs,
                    total_steps, init_steps,
                    exit_rate, nll_weight,
                    step_type, x_dim, y_dim,
                    att_spec_dim, glimpse_dim, glimpse_count,
                    reader_mlp, writer_mlp,
                    con_mlp_in, con_rnn, con_mlp_out,
                    rav_mlp_in, rav_rnn, rav_mlp_out,
                    att_mlp_in,
                    **kwargs):
        super(SeqCondGen2dSL, self).__init__(**kwargs)
        if not ((step_type == 'add') or (step_type == 'jump')):
            raise ValueError('step_type must be jump or add')
        # record basic structural parameters
        self.x_and_y_are_seqs = x_and_y_are_seqs
        self.total_steps = total_steps
        self.init_steps = init_steps
        self.exit_rate = exit_rate
        self.nll_weight = nll_weight
        self.step_type = step_type
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.att_spec_dim = att_spec_dim
        self.all_spec_dim = glimpse_count * att_spec_dim
        self.glimpse_dim = glimpse_dim
        self.glimpse_count = glimpse_count
        # construct a sequence of scales for measuring NLL. we'll use scales
        # corresponding to some fixed number of guaranteed steps, followed by
        # a constant probability of early stopping.
        self.nll_scales = self._construct_nll_scales()

        # grab handles for shared read/write models
        self.reader_mlp = reader_mlp
        self.writer_mlp = writer_mlp
        # grab handles for sequential stochastic observe and control models
        self.con_mlp_in = con_mlp_in
        self.con_rnn = con_rnn
        self.con_mlp_out = con_mlp_out
        self.rav_mlp_in = rav_mlp_in
        self.rav_rnn = rav_rnn
        self.rav_mlp_out = rav_mlp_out
        self.att_mlp_in = att_mlp_in

        # set up the transform from belief state to observation space
        self.c_to_y = tensor.nnet.sigmoid

        # create a shared variable switch for controlling sampling
        ones_ary = numpy.ones((1,)).astype(theano.config.floatX)
        self.train_switch = theano.shared(value=ones_ary, name='train_switch')
        # create shared variables for controlling KLd terms
        self.lam_kld_q2p = theano.shared(value=ones_ary, name='lam_kld_q2p')
        self.lam_kld_p2q = theano.shared(value=ones_ary, name='lam_kld_p2q')
        self.set_lam_kld(lam_kld_q2p=0.95, lam_kld_p2q=0.05)
        # set up weight for controlling influence of attention placement term
        self.lam_nll_att = theano.shared(value=ones_ary, name='lam_nll_att')
        self.set_lam_nll_att(lam_nll_att=10.0)
        # create shared variables for controlling optimization/updates
        self.lr = theano.shared(value=0.0001*ones_ary, name='lr')
        self.mom_1 = theano.shared(value=0.9*ones_ary, name='mom_1')
        self.mom_2 = theano.shared(value=0.99*ones_ary, name='mom_2')

        # setup a "null pointer" that will point to the computation graph
        # for this model, which can be built by self.build_model_funcs()...
        self.cg = None

        # record the sub-models around which this model is built
        self.children = [self.reader_mlp, self.writer_mlp,
                         self.con_mlp_in, self.con_rnn, self.con_mlp_out,
                         self.rav_mlp_in, self.rav_rnn, self.rav_mlp_out,
                         self.att_mlp_in]
        return

    def set_sgd_params(self, lr=0.01, mom_1=0.9, mom_2=0.999):
        """
        Set learning rate and momentum parameter for all updates.
        """
        zero_ary = numpy.zeros((1,))
        # set learning rate
        new_lr = zero_ary + lr
        self.lr.set_value(to_fX(new_lr))
        # set momentums (use first and second order "momentum")
        new_mom_1 = zero_ary + mom_1
        self.mom_1.set_value(to_fX(new_mom_1))
        new_mom_2 = zero_ary + mom_2
        self.mom_2.set_value(to_fX(new_mom_2))
        return

    def set_lam_kld(self, lam_kld_q2p=0.0, lam_kld_p2q=1.0):
        """
        Set the relative weight of various KL-divergence terms.

        kld_q2p: KLd between guide and primary controller. KL(q || p)
        kld_p2q: KLd between primary and guide controller. KL(p || q)
        """
        zero_ary = numpy.zeros((1,))
        new_lam = zero_ary + lam_kld_q2p
        self.lam_kld_q2p.set_value(to_fX(new_lam))
        new_lam = zero_ary + lam_kld_p2q
        self.lam_kld_p2q.set_value(to_fX(new_lam))
        return

    def set_lam_nll_att(self, lam_nll_att=1.0):
        """
        Set the relative weight of attention placement term.
        """
        zero_ary = numpy.zeros((1,))
        new_lam = zero_ary + lam_nll_att
        self.lam_nll_att.set_value(to_fX(new_lam))
        return

    def _construct_nll_scales(self):
        """
        Construct a sequence of scales for weighting NLL measurements.
        """
        nll_scales = shared_floatx_nans((self.total_steps,), name='nll_scales')
        if self.x_and_y_are_seqs:
            # construct NLL scales for each step, when using sequential x/y
            np_scales = self.nll_weight * numpy.ones((self.total_steps,))
            np_scales[0:self.init_steps] = 0.0
        else:
            # construct NLL scales for each step, when using static x/y
            np_scales = numpy.zeros((self.total_steps,))
            prob_to_get_here = 1.0
            for i in range(self.total_steps):
                if ((i+1) > self.init_steps):
                    np_scales[i] = prob_to_get_here * self.exit_rate
                    prob_to_get_here = prob_to_get_here * (1.0 - self.exit_rate)
            # force exit on the last step -- i.e. assign it any missing weight
            sum_of_scales = numpy.sum(np_scales)
            missing_weight = 1.0 - sum_of_scales
            np_scales[-1] = np_scales[-1] + missing_weight
        nll_scales.set_value(np_scales.astype(theano.config.floatX))
        return nll_scales

    def _allocate(self):
        """
        Allocate shared parameters used by this model.
        """
        # get size information for the desired parameters
        c_dim = self.get_dim('c')
        cc_dim = self.get_dim('c_con')
        hc_dim = self.get_dim('h_con')
        cr_dim = self.get_dim('c_rav')
        hr_dim = self.get_dim('h_rav')
        # self.c_0 provides initial belief state
        self.c_0 = shared_floatx_nans((1,c_dim), name='c_0')
        add_role(self.c_0, PARAMETER)
        # self.cc_0/self.hc_0 provides initial state of the controller
        self.cc_0 = shared_floatx_nans((1,cc_dim), name='cc_0')
        add_role(self.cc_0, PARAMETER)
        self.hc_0 = shared_floatx_nans((1,hc_dim), name='hc_0')
        add_role(self.hc_0, PARAMETER)
        # self.cr_0/self.hr_0 provides initial state of the variational controller
        self.cr_0 = shared_floatx_nans((1,cr_dim), name='cr_0')
        add_role(self.cr_0, PARAMETER)
        self.hr_0 = shared_floatx_nans((1,hr_dim), name='hr_0')
        add_role(self.hr_0, PARAMETER)
        # add the theano shared variables to our parameter lists
        self.params.extend([ self.c_0, \
                             self.cc_0, self.cr_0, \
                             self.hc_0, self.hr_0 ])
        return

    def _initialize(self):
        # initialize all parameters to zeros...
        for p in self.params:
            p_nan = p.get_value(borrow=False)
            p_zeros = numpy.zeros(p_nan.shape)
            p.set_value(p_zeros.astype(theano.config.floatX))
        return

    def get_dim(self, name):
        if name == 'x':
            return self.x_dim
        elif name == 'nll_scale':
            return 1
        elif name == 'y_att':
            return 2
        elif name in ['u', 'z']:
            return self.con_mlp_out.output_dim
        elif name in ['u_att']:
            return self.all_spec_dim
        elif name in ['z_att']:
            return self.att_mlp_in.input_dim
        elif name in ['c_as_y', 'y']:
            return self.y_dim
        elif name == 'c':
            return self.writer_mlp.output_dim
        elif name == 'h_con':
            return self.con_rnn.get_dim('states')
        elif name == 'c_con':
            return self.con_rnn.get_dim('cells')
        elif name == 'h_rav':
            return self.rav_rnn.get_dim('states')
        elif name == 'c_rav':
            return self.rav_rnn.get_dim('cells')
        elif name in ['nll', 'kl_q2p', 'kl_p2q']:
            return 0
        else:
            super(SeqCondGen2dSL, self).get_dim(name)
        return

    #------------------------------------------------------------------------

    @recurrent(sequences=['x', 'y', 'y_att', 'u', 'u_att', 'nll_scale'], contexts=[],
               states=['c', 'h_con', 'c_con', 'h_rav', 'c_rav'],
               outputs=['c', 'h_con', 'c_con', 'h_rav', 'c_rav', 'c_as_y', 'nll', 'nll_att', 'kl_q2p', 'kl_p2q', 'att_map', 'read_img'])
    def iterate(self, x, y, y_att, u, u_att, nll_scale, c, h_con, c_con, h_rav, c_rav):
        # Get the current prediction for y
        if self.step_type == 'jump':
            # Controller hidden state tracks belief state (for jump steps)
            c = self.writer_mlp.apply(c_con)
        # Convert from belief state to prediction for y
        c_as_y = self.c_to_y(c)

        ######################################
        # Update the guide controller state. #
        ######################################
        #
        # Guide controller can condition on any available information.
        #
        nll_grad_rav = y - c_as_y # condition on NLL gradient information
        i_rav = self.rav_mlp_in.apply(tensor.concatenate( \
                                      [h_con, nll_grad_rav], axis=1))
        h_rav, c_rav = self.rav_rnn.apply(states=h_rav, cells=c_rav, \
                                          inputs=i_rav, iterate=False)

        #################################################################
        # Convert guide and primary controller state into samples of z. #
        #################################################################
        #
        # In this model, z contains two subsets: z_att and z_slf. The values in
        # z_slf are fed back into the main controller LSTM as inputs, to provide
        # stochasticity beyond that which stems from uncertainty in the read
        # operation. The values in z_att are fed into self.att_mlp_in, which
        # transforms them into a set of self.glimpse_count specs for applying
        # the attention-based reader module.
        #
        # By representing stochasticity in the attention placement using a set
        # of latent variables, rather than directly specifying wobble in the
        # attention placement, we can form multi-modal distributions over the
        # attention placement at each step. This multi-modality might be useful
        # when the model must divide its attention among multiple objects.
        #
        p_z_mean, p_z_logvar, p_z = \
                self.con_mlp_out.apply(h_con, u)
        q_z_mean, q_z_logvar, q_z = \
                self.rav_mlp_out.apply(tensor.concatenate([h_rav, 2.0*y_att],axis=1), u)
        # mix samples from p/q based on value of self.train_switch
        z = (self.train_switch[0] * q_z) + \
            ((1.0 - self.train_switch[0]) * p_z)
        # compute KL(q || p) and KL(p || q) for latent var distributions
        kl_q2p = tensor.sum(gaussian_kld(q_z_mean, q_z_logvar, \
                            p_z_mean, p_z_logvar), axis=1)
        kl_p2q = tensor.sum(gaussian_kld(p_z_mean, p_z_logvar, \
                            q_z_mean, q_z_logvar), axis=1)
        ##################################################
        # Split z into attention and self-control parts. #
        ##################################################
        z_att_dim = self.get_dim('z_att')
        z_att = z[:,:z_att_dim]
        z_slf = z[:,z_att_dim:]
        # transform latent variables z_att into one or more attention specs
        as_mean, as_logvar, att_specs = self.att_mlp_in.apply(z_att, u_att)

        #######################################################
        # Apply the attention-based reader to the input in x. #
        # -- This may be done for one or many glimpses. --    #
        #######################################################
        if self.glimpse_count == 1:
            # shortcut for single glimpse setting
            read_out = self.reader_mlp.read(x, x, att_specs)
            att_map = self.reader_mlp.att_map(att_specs)
            read_img = self.reader_mlp.write(read_out, att_specs)
            att_loc = att_specs[:,0:2]
            att_res = att_loc - y_att
            nll_att = nll_scale * tensor.sum(att_res**2.0, axis=1)
        else:
            # compute the outcomes of multiple simultaneous glimpses
            read_outs = []
            att_maps = []
            read_imgs = []
            att_locs = []
            for g_num in range(self.glimpse_count):
                as_start = g_num * self.att_spec_dim
                as_end = (g_num + 1) * self.att_spec_dim
                as_i = att_specs[:,as_start:as_end]
                ro_i = self.reader_mlp.read(x, x, as_i)
                read_outs.append(ro_i)
                att_maps.append(self.reader_mlp.att_map(as_i))
                read_imgs.append(self.reader_mlp.write(ro_i, as_i))
                att_locs.append(as_i[:,0:2])
            read_out = tensor.concatenate(read_outs, axis=1)
            att_map = tensor.concatenate(att_maps, axis=1)
            read_img = tensor.concatenate(read_imgs, axis=1)
            att_loc = tensor.stack(att_locs, axis=2)
            att_res = y_att.dimshuffle(0,1,'x') - att_loc
            nll_att_all = tensor.sum(att_res**2.0, axis=1)
            nll_att = nll_scale * tensor.min(nll_att_all, axis=1)

        #############################################
        # Update the primary controller LSTM state. #
        #############################################
        #
        # The primary controller gets deterministic and stochastic inputs from
        # its prior self at each time step. Deterministic input is its visible
        # state from the previous step. The stochastic inputs are z_slf, which
        # provides a "direct" noisy channel from t -> t+1, and read_out and
        # att_specs. read_out and att_specs are the read-outs and placement
        # specs for multiple applications of the attention module.
        #
        # The primary controller's visible state "h_con" sets distributions
        # over z, which gets split into z_slf and z_att. z_att is transformed
        # into one or more attention specs by self.att_mlp_in.
        #
        i_con = self.con_mlp_in.apply(tensor.concatenate([z_slf, read_out, att_specs], axis=1))
        h_con, c_con = self.con_rnn.apply(states=h_con, cells=c_con, \
                                          inputs=i_con, iterate=False)

        ###############################################################
        # Update the "workspace" (maintained either in c or in c_con) #
        ###############################################################
        if self.step_type == 'add':
            # Controller visible state dictates belief updates (for add steps)
            c = c + self.writer_mlp.apply(h_con)
        else:
            # Controller hidden state tracks belief state (for jump steps)
            c = self.writer_mlp.apply(c_con)
        # Convert from belief state to prediction for y
        c_as_y = self.c_to_y(c)

        ########################################################
        # Compute the NLL of the y prediction as of this step. #
        ########################################################
        nll = -nll_scale * tensor.flatten(log_prob_bernoulli(y, c_as_y))
        return c, h_con, c_con, h_rav, c_rav, c_as_y, nll, nll_att, kl_q2p, kl_p2q, att_map, read_img

    #------------------------------------------------------------------------

    @application(inputs=['x', 'y', 'y_att'],
                 outputs=['cs', 'c_as_ys', 'nlls', 'nll_atts', 'kl_q2ps', 'kl_p2qs', 'att_maps', 'read_imgs'])
    def process_inputs(self, x, y, y_att):
        # get important size and shape information
        z_dim = self.get_dim('z')
        cc_dim = self.get_dim('c_con')
        cr_dim = self.get_dim('c_rav')
        hc_dim = self.get_dim('h_con')
        hr_dim = self.get_dim('h_rav')
        as_dim = self.all_spec_dim

        if self.x_and_y_are_seqs:
            batch_size = x.shape[1]
        else:
            assert False, "x and y must be sequences"

        # get initial states for all model components
        c0 = self.c_0.repeat(batch_size, axis=0)
        cc0 = self.cc_0.repeat(batch_size, axis=0)
        hc0 = self.hc_0.repeat(batch_size, axis=0)
        cr0 = self.cr_0.repeat(batch_size, axis=0)
        hr0 = self.hr_0.repeat(batch_size, axis=0)

        # get noise samples for main stochastic variables
        u = self.theano_rng.normal(
                    size=(self.total_steps, batch_size, z_dim),
                    avg=0., std=1.)
        # get noise samples for ultimate attention wobble
        u_att = 0.25 * self.theano_rng.normal(
                    size=(self.total_steps, batch_size, as_dim),
                    avg=0., std=1.)

        # run the multi-stage guided generative process
        cs, _, _, _, _, c_as_ys, nlls, nll_atts, kl_q2ps, kl_p2qs, att_maps, read_imgs = \
                self.iterate(x=x, y=y, y_att=y_att, u=u, u_att=u_att,
                             nll_scale=self.nll_scales, c=c0,
                             h_con=hc0, c_con=cc0, h_rav=hr0, c_rav=cr0)

        # add name tags to the constructed values
        cs.name = "cs"
        c_as_ys.name = "c_as_ys"
        nlls.name = "nlls"
        nll_atts.name = "nll_atts"
        kl_q2ps.name = "kl_q2ps"
        kl_p2qs.name = "kl_p2qs"
        att_maps.name = "att_maps"
        read_imgs.name = "read_imgs"
        return cs, c_as_ys, nlls, nll_atts, kl_q2ps, kl_p2qs, att_maps, read_imgs

    def build_model_funcs(self):
        """
        Build the symbolic costs and theano functions relevant to this model.
        """
        # some symbolic vars to represent various inputs/outputs
        if self.x_and_y_are_seqs:
            x_sym = tensor.tensor3('x_sym')
            y_sym = tensor.tensor3('y_sym')
            y_att_sym = tensor.tensor3('y_att_sym')
        else:
            assert False, "x and y must be sequences"

        # collect estimates of y given x produced by this model
        cs, c_as_ys, nlls, nll_atts, kl_q2ps, kl_p2qs, att_maps, read_imgs = \
                self.process_inputs(x_sym, y_sym, y_att_sym)

        # get the expected NLL part of the VFE bound
        self.nll_term = nlls.sum(axis=0).mean()
        self.nll_term.name = "nll_term"

        # get the expected NLL attention part of the VFE bound
        self.nll_att_term = self.lam_nll_att[0] * nll_atts.sum(axis=0).mean()
        self.nll_att_term.name = "nll_atts_term"

        # get KL(q || p) and KL(p || q)
        self.kld_q2p_term = self.lam_kld_q2p[0] * kl_q2ps.sum(axis=0).mean()
        self.kld_q2p_term.name = "kld_q2p_term"
        self.kld_p2q_term = self.lam_kld_p2q[0] * kl_p2qs.sum(axis=0).mean()
        self.kld_p2q_term.name = "kld_p2q_term"

        # compute a cost that depends on all trainable model parameters
        partial_cost = self.nll_term + \
                       self.nll_att_term + \
                       self.kld_q2p_term + \
                       self.kld_p2q_term

        # grab handles for all the optimizable parameters in our cost
        self.cg = ComputationGraph([partial_cost])
        self.joint_params = self.get_model_params(ary_type='theano')

        # apply some l2 regularization to the model parameters
        self.reg_term = (1e-5 * sum([tensor.sum(p**2.0) for p in self.joint_params]))
        self.reg_term.name = "reg_term"

        # compute the full cost w.r.t. which we will optimize params
        self.joint_cost = self.nll_term + \
                          self.nll_att_term + \
                          self.kld_q2p_term + \
                          self.kld_p2q_term + \
                          self.reg_term
        self.joint_cost.name = "joint_cost"

        # get the gradient of the joint cost for all optimizable parameters
        print("Computing gradients of joint_cost...")
        self.joint_grads = OrderedDict()
        grad_list = tensor.grad(self.joint_cost, self.joint_params)
        for i, p in enumerate(self.joint_params):
            self.joint_grads[p] = grad_list[i]

        # construct the updates for all trainable parameters
        self.joint_updates = get_adam_updates(params=self.joint_params,
                grads=self.joint_grads, alpha=self.lr,
                beta1=self.mom_1, beta2=self.mom_2,
                mom2_init=1e-3, smoothing=1e-5, max_grad_norm=10.0)

        # collect the outputs to return from this function
        outputs = [self.joint_cost, self.nll_term, self.nll_att_term, \
                   self.kld_q2p_term, self.kld_p2q_term, self.reg_term]
        # collect the required inputs
        inputs = [x_sym, y_sym, y_att_sym]

        # compile the theano functions for computing stuff, like for real
        print("Compiling model training/update function...")
        self.train_joint = theano.function(inputs=inputs, \
                                           outputs=outputs, \
                                           updates=self.joint_updates)
        print("Compiling NLL bound estimator function...")
        self.compute_nll_bound = theano.function(inputs=inputs, \
                                                 outputs=outputs)
        return

    def build_attention_funcs(self):
        """
        Build functions for visualizing the behavior of this model, assuming
        self.reader_mlp is SimpleAttentionReader2d or SimpleAttentionReader1d.
        """
        # some symbolic vars to represent various inputs/outputs
        if self.x_and_y_are_seqs:
            x_sym = tensor.tensor3('x_sym_att_funcs')
            y_sym = tensor.tensor3('y_sym_att_funcs')
            y_att_sym = tensor.tensor3('y_att_sym_att_funcs')
        else:
            assert False, "x and y must be sequences"
        # collect estimates of y given x produced by this model
        cs, c_as_ys, nlls, nll_atts, kl_q2ps, kl_p2qs, att_maps, read_imgs = \
                self.process_inputs(x_sym, y_sym, y_att_sym)

        # build the function for computing the attention trajectories
        print("Compiling attention tracker...")
        inputs = [x_sym, y_sym, y_att_sym]
        outputs = [c_as_ys, att_maps, read_imgs]
        sample_func = theano.function(inputs=inputs, \
                                      outputs=outputs)
        def switchy_sampler(x, y, y_att, sample_source='q'):
            # store value of sample source switch, to restore later
            old_switch = self.train_switch.get_value()
            if sample_source == 'p':
                # take samples from the primary policy
                zeros_ary = numpy.zeros((1,)).astype(theano.config.floatX)
                self.train_switch.set_value(zeros_ary)
            else:
                # take samples from the guide policy
                ones_ary = numpy.ones((1,)).astype(theano.config.floatX)
                self.train_switch.set_value(ones_ary)
            # sample prediction and attention trajectories
            outs = sample_func(x, y, y_att)
            # set sample source switch back to previous value
            self.train_switch.set_value(old_switch)
            # grab prediction values
            y_preds = outs[0]
            # aggregate sequential attention maps
            obs_dim = self.x_dim
            seq_len = self.total_steps
            samp_count = outs[1].shape[1]
            att_count = self.glimpse_count
            a_maps = numpy.zeros((outs[1].shape[0], outs[1].shape[1], obs_dim))
            for a_num in range(att_count):
                start_idx = a_num * obs_dim
                end_idx = (a_num + 1) * obs_dim
                for s1 in range(samp_count):
                    for s2 in range(seq_len):
                        a_maps[s2,s1,:] = a_maps[s2,s1,:] + \
                                          outs[1][s2,s1,start_idx:end_idx]
            # aggregate sequential attention read-outs
            r_outs = numpy.zeros((outs[2].shape[0], outs[2].shape[1], obs_dim))
            for a_num in range(att_count):
                start_idx = a_num * obs_dim
                end_idx = (a_num + 1) * obs_dim
                for s1 in range(samp_count):
                    for s2 in range(seq_len):
                        r_outs[s2,s1,:] = r_outs[s2,s1,:] + \
                                          outs[2][s2,s1,start_idx:end_idx]
            result = [y_preds, a_maps, r_outs]
            return result
        self.sample_attention = switchy_sampler
        return

    def get_model_params(self, ary_type='numpy'):
        """
        Get the optimizable parameters in this model. This returns a list
        and, to reload this model's parameters, the list must stay in order.

        This can provide shared variables or numpy arrays.
        """
        if self.cg is None:
            self.build_model_funcs()
        joint_params = VariableFilter(roles=[PARAMETER])(self.cg.variables)
        if ary_type == 'numpy':
            for i, p in enumerate(joint_params):
                joint_params[i] = p.get_value(borrow=False)
        return joint_params

    def set_model_params(self, numpy_param_list):
        """
        Set the optimizable parameters in this model. This requires a list
        and, to reload this model's parameters, the list must be in order.
        """
        if self.cg is None:
            self.build_model_funcs()
        # grab handles for all the optimizable parameters in our cost
        joint_params = VariableFilter(roles=[PARAMETER])(self.cg.variables)
        for i, p in enumerate(joint_params):
            joint_params[i].set_value(to_fX(numpy_param_list[i]))
        return joint_params

    def save_model_params(self, f_name=None):
        """
        Save model parameters to a pickle file, in numpy form.
        """
        numpy_params = self.get_model_params(ary_type='numpy')
        f_handle = file(f_name, 'wb')
        # dump the dict self.params, which just holds "simple" python values
        cPickle.dump(numpy_params, f_handle, protocol=-1)
        f_handle.close()
        return

    def load_model_params(self, f_name=None):
        """
        Load model parameters from a pickle file, in numpy form.
        """
        pickle_file = open(f_name)
        numpy_params = cPickle.load(pickle_file)
        self.set_model_params(numpy_params)
        pickle_file.close()
        return

############################################################
############################################################
## ATTENTION-BASED PERCEPTION UNDER TIME CONSTRAINTS      ##
############################################################
############################################################

class SeqCondGenX(BaseRecurrent, Initializable, Random):
    """
    SeqCondGenX -- constructs conditional densities under time constraints.

    This model sequentially constructs a conditional density estimate by taking
    repeated glimpses at the input x, and constructing a hypothesis about the
    output y. The objective is maximum likelihood for (x,y) pairs drawn from
    some training set. We learn a proper generative model, using variational
    inference -- which can be interpreted as a sort of guided policy search.

    The input pairs (x, y) can be either "static" or "sequential". In the
    static case, the same x and y are used at every step of the hypothesis
    construction loop. In the sequential case, x and y can change at each step
    of the loop.

    Parameters:
        x_and_y_are_seqs: boolean telling whether the conditioning information
                          and prediction targets are sequential.
        total_steps: total number of steps in sequential estimation process
        init_steps: number of steps prior to first NLL measurement
        exit_rate: probability of exiting following each non "init" step
                   **^^ THIS IS SET TO 0 WHEN USING SEQUENTIAL INPUT ^^**
        nll_weight: weight for the prediction NLL term at each step.
                   **^^ THIS IS IGNORED WHEN USING STATIC INPUT ^^**
        step_type: whether to use "additive" steps or "jump" steps
                   -- jump steps predict directly from the controller LSTM's
                      "hidden" state (a.k.a. its memory cells).
        x_dim: dimension of inputs on which to condition
        y_dim: dimension of outputs to predict
        reader_mlp: used for reading from the input
        writer_mlp: used for writing to the output prediction
        con_mlp_in: preprocesses input to the "controller" LSTM
        con_rnn: the "controller" LSTM
        con_mlp_out: CondNet for distribution over att spec given con_rnn
        gen_mlp_in: preprocesses input to the "generator" LSTM
        gen_rnn: the "generator" LSTM
        gen_mlp_out: CondNet for distribution over z given gen_rnn
        var_mlp_in: preprocesses input to the "variational" LSTM
        var_rnn: the "variational" LSTM
        var_mlp_out: CondNet for distribution over z given gen_rnn
    """
    def __init__(self, x_and_y_are_seqs,
                    total_steps, init_steps,
                    exit_rate, nll_weight,
                    step_type, x_dim, y_dim,
                    reader_mlp, writer_mlp,
                    con_mlp_in, con_rnn, con_mlp_out,
                    gen_mlp_in, gen_rnn, gen_mlp_out,
                    var_mlp_in, var_rnn, var_mlp_out,
                    **kwargs):
        super(SeqCondGenX, self).__init__(**kwargs)
        if not ((step_type == 'add') or (step_type == 'jump')):
            raise ValueError('step_type must be jump or add')
        # record basic structural parameters
        self.x_and_y_are_seqs = x_and_y_are_seqs
        self.total_steps = total_steps
        self.init_steps = init_steps
        self.exit_rate = exit_rate
        self.nll_weight = nll_weight
        self.step_type = step_type
        self.x_dim = x_dim
        self.y_dim = y_dim
        # construct a sequence of scales for measuring NLL. we'll use scales
        # corresponding to some fixed number of guaranteed steps, followed by
        # a constant probability of early stopping. any "residual" probability
        # will be picked up by the final step.
        self.nll_scales = self._construct_nll_scales()
        # grab handles for shared read/write models
        self.reader_mlp = reader_mlp
        self.writer_mlp = writer_mlp
        # dimension for attention specification (i.e. location, scale, etc.)
        self.att_spec_dim = 5
        # grab handles for sequential read/write models
        self.con_mlp_in = con_mlp_in
        self.con_rnn = con_rnn
        self.con_mlp_out = con_mlp_out
        self.gen_mlp_in = gen_mlp_in
        self.gen_rnn = gen_rnn
        self.gen_mlp_out = gen_mlp_out
        self.var_mlp_in = var_mlp_in
        self.var_rnn = var_rnn
        self.var_mlp_out = var_mlp_out
        # create a shared variable switch for controlling sampling
        ones_ary = numpy.ones((1,)).astype(theano.config.floatX)
        self.train_switch = theano.shared(value=ones_ary, name='train_switch')
        # create shared variables for controlling KLd terms
        self.lam_kld_q2p = theano.shared(value=ones_ary, name='lam_kld_q2p')
        self.lam_kld_p2q = theano.shared(value=ones_ary, name='lam_kld_p2q')
        self.set_lam_kld(lam_kld_q2p=0.95, lam_kld_p2q=0.05)
        # create shared variables for controlling optimization/updates
        self.lr = theano.shared(value=0.0001*ones_ary, name='lr')
        self.mom_1 = theano.shared(value=0.9*ones_ary, name='mom_1')
        self.mom_2 = theano.shared(value=0.99*ones_ary, name='mom_2')

        # set noise scale for the attention placement
        self.att_noise = 0.1

        # setup a "null pointer" that will point to the computation graph
        # for this model, which can be built by self.build_model_funcs()...
        self.cg = None

        # record the sub-models around which this model is built
        self.children = [self.reader_mlp, self.writer_mlp,
                         self.con_mlp_in, self.con_rnn, self.con_mlp_out,
                         self.gen_mlp_in, self.gen_rnn, self.gen_mlp_out,
                         self.var_mlp_in, self.var_rnn, self.var_mlp_out]
        return

    def set_sgd_params(self, lr=0.01, mom_1=0.9, mom_2=0.999):
        """
        Set learning rate and momentum parameter for all updates.
        """
        zero_ary = numpy.zeros((1,))
        # set learning rate
        new_lr = zero_ary + lr
        self.lr.set_value(to_fX(new_lr))
        # set momentums (use first and second order "momentum")
        new_mom_1 = zero_ary + mom_1
        self.mom_1.set_value(to_fX(new_mom_1))
        new_mom_2 = zero_ary + mom_2
        self.mom_2.set_value(to_fX(new_mom_2))
        return

    def set_lam_kld(self, lam_kld_q2p=0.0, lam_kld_p2q=1.0):
        """
        Set the relative weight of various KL-divergence terms.

        kld_q2p: KLd between guide reader and primary reader. KL(q||p)
        kld_p2q: KLd between primary reader and guide reader. KL(p||q)
        """
        zero_ary = numpy.zeros((1,))
        new_lam = zero_ary + lam_kld_q2p
        self.lam_kld_q2p.set_value(to_fX(new_lam))
        new_lam = zero_ary + lam_kld_p2q
        self.lam_kld_p2q.set_value(to_fX(new_lam))
        return

    def _construct_nll_scales(self):
        """
        Construct a sequence of scales for weighting NLL measurements.
        """
        nll_scales = shared_floatx_nans((self.total_steps,), name='nll_scales')
        if self.x_and_y_are_seqs:
            # construct NLL scales for each step, when using sequential x/y
            np_scales = self.nll_weight * numpy.ones((self.total_steps,))
            np_scales[0:self.init_steps] = 0.0
        else:
            # construct NLL scales for each step, when using static x/y
            np_scales = numpy.zeros((self.total_steps,))
            prob_to_get_here = 1.0
            for i in range(self.total_steps):
                if ((i+1) > self.init_steps):
                    np_scales[i] = prob_to_get_here * self.exit_rate
                    prob_to_get_here = prob_to_get_here * (1.0 - self.exit_rate)
            # force exit on the last step -- i.e. assign it any missing weight
            sum_of_scales = numpy.sum(np_scales)
            missing_weight = 1.0 - sum_of_scales
            np_scales[-1] = np_scales[-1] + missing_weight
        nll_scales.set_value(np_scales.astype(theano.config.floatX))
        return nll_scales

    def _allocate(self):
        """
        Allocate shared parameters used by this model.
        """
        # get size information for the desired parameters
        c_dim = self.get_dim('c')
        cc_dim = self.get_dim('c_con')
        hc_dim = self.get_dim('h_con')
        cg_dim = self.get_dim('c_gen')
        hg_dim = self.get_dim('h_gen')
        cv_dim = self.get_dim('c_var')
        hv_dim = self.get_dim('h_var')
        # self.c_0 provides initial state of the next column prediction
        self.c_0 = shared_floatx_nans((1,c_dim), name='c_0')
        add_role(self.c_0, PARAMETER)
        # self.cc_0/self.hc_0 provides initial state of the controller
        self.cc_0 = shared_floatx_nans((1,cc_dim), name='cc_0')
        add_role(self.cc_0, PARAMETER)
        self.hc_0 = shared_floatx_nans((1,hc_dim), name='hc_0')
        add_role(self.hc_0, PARAMETER)
        # self.cg_0/self.hg_0 provides initial state of the primary policy
        self.cg_0 = shared_floatx_nans((1,cg_dim), name='cg_0')
        add_role(self.cg_0, PARAMETER)
        self.hg_0 = shared_floatx_nans((1,hg_dim), name='hg_0')
        add_role(self.hg_0, PARAMETER)
        # self.cv_0/self.hv_0 provides initial state of the guide policy
        self.cv_0 = shared_floatx_nans((1,cv_dim), name='cv_0')
        add_role(self.cv_0, PARAMETER)
        self.hv_0 = shared_floatx_nans((1,hv_dim), name='hv_0')
        add_role(self.hv_0, PARAMETER)
        # add the theano shared variables to our parameter lists
        self.params.extend([ self.c_0, \
                             self.cc_0, self.cg_0, self.cv_0, \
                             self.hc_0, self.hg_0, self.hv_0 ])
        return

    def _initialize(self):
        # initialize all parameters to zeros...
        for p in self.params:
            p_nan = p.get_value(borrow=False)
            p_zeros = numpy.zeros(p_nan.shape)
            p.set_value(p_zeros.astype(theano.config.floatX))
        return

    def get_dim(self, name):
        if name == 'x':
            return self.x_dim
        elif name == 'nll_scale':
            return 1
        elif name in ['c', 'c0', 'y']:
            return self.y_dim
        elif name in ['u', 'z']:
            return self.gen_mlp_out.get_dim('output')
        elif name in ['u_att']:
            return self.att_spec_dim
        elif name in ['h_con', 'hc0']:
            return self.con_rnn.get_dim('states')
        elif name == 'c_con':
            return self.con_rnn.get_dim('cells')
        elif name == 'h_gen':
            return self.gen_rnn.get_dim('states')
        elif name == 'c_gen':
            return self.gen_rnn.get_dim('cells')
        elif name == 'h_var':
            return self.var_rnn.get_dim('states')
        elif name == 'c_var':
            return self.var_rnn.get_dim('cells')
        elif name in ['nll', 'kl_q2p', 'kl_p2q']:
            return 0
        else:
            super(SeqCondGenX, self).get_dim(name)
        return

    #------------------------------------------------------------------------

    @recurrent(sequences=['x', 'y', 'u', 'u_att', 'nll_scale'], contexts=[],
               states=['c', 'h_con', 'c_con', 'h_gen', 'c_gen', 'h_var', 'c_var'],
               outputs=['c', 'h_con', 'c_con', 'h_gen', 'c_gen', 'h_var', 'c_var', 'c_as_y', 'nll', 'kl_q2p', 'kl_p2q', 'att_map', 'read_img'])
    def iterate(self, x, y, u, u_att, nll_scale, c, h_con, c_con, h_gen, c_gen, h_var, c_var):
        if self.step_type == 'add':
            # additive steps use c as a "direct workspace", which means it's
            # already directly comparable to y.
            c_as_y = tensor.nnet.sigmoid(c)
        else:
            # non-additive steps use c_con as a "latent workspace", which means
            # it needs to be transformed before being comparable to y.
            c_as_y = tensor.nnet.sigmoid(self.writer_mlp.apply(c_con))

        # estimate conditional over attention spec given h_con (from time t-1)
        as_mean, as_logvar, att_spec = \
                self.con_mlp_out.apply(h_con, u_att)

        # apply the attention-based reader to the input in x
        read_out = self.reader_mlp.apply(x, x, att_spec)
        att_map = self.reader_mlp.att_map(att_spec)
        read_img = self.reader_mlp.write(read_out, att_spec)

        # update the primary RNN state
        i_gen = self.gen_mlp_in.apply( \
                tensor.concatenate([read_out, att_spec, h_con], axis=1))
        h_gen, c_gen = self.gen_rnn.apply(states=h_gen, cells=c_gen,
                                          inputs=i_gen, iterate=False)
        # update the guide RNN state
        nll_grad = y - c_as_y # condition on NLL gradient information
        i_var = self.var_mlp_in.apply( \
                tensor.concatenate([nll_grad, read_out, att_spec, h_con], axis=1))
        h_var, c_var = self.var_rnn.apply(states=h_var, cells=c_var,
                                          inputs=i_var, iterate=False)

        # estimate primary conditional over z given h_gen
        p_z_mean, p_z_logvar, p_z = \
                self.gen_mlp_out.apply(h_gen, u)
        # estimate guide conditional over z given h_var
        q_z_mean, q_z_logvar, q_z = \
                self.var_mlp_out.apply(h_var, u)

        # mix samples from p/q based on value of self.train_switch
        z = (self.train_switch[0] * q_z) + \
            ((1.0 - self.train_switch[0]) * p_z)

        # update the controller RNN state, using the sampled z values
        #i_con = self.con_mlp_in.apply(tensor.concatenate([z], axis=1))
        i_con = self.con_mlp_in.apply(z)
        h_con, c_con = self.con_rnn.apply(states=h_con, cells=c_con, \
                                          inputs=i_con, iterate=False)

        # update the "workspace" (stored in c)
        if self.step_type == 'add':
            c = c + self.writer_mlp.apply(h_con)
        else:
            c = self.writer_mlp.apply(c_con)

        # compute the NLL of the reconstruction as of this step. the NLL at
        # each step is rescaled by a factor such that the sum of the factors
        # for all steps is 1, and all factors are non-negative.
        c_as_y = tensor.nnet.sigmoid(c)
        nll = -nll_scale * tensor.flatten(log_prob_bernoulli(y, c_as_y))
        # compute KL(q || p) and KL(p || q) for this step
        kl_q2p = tensor.sum(gaussian_kld(q_z_mean, q_z_logvar, \
                            p_z_mean, p_z_logvar), axis=1)
        kl_p2q = tensor.sum(gaussian_kld(p_z_mean, p_z_logvar, \
                            q_z_mean, q_z_logvar), axis=1)
        return c, h_con, c_con, h_gen, c_gen, h_var, c_var, c_as_y, nll, kl_q2p, kl_p2q, att_map, read_img

    #------------------------------------------------------------------------

    @application(inputs=['x', 'y'],
                 outputs=['xs', 'cs', 'h_cons', 'c_as_ys', 'nlls', 'kl_q2ps', 'kl_p2qs', 'att_maps', 'read_imgs'])
    def process_inputs(self, x, y):
        # get important size and shape information

        z_dim = self.get_dim('z')
        cc_dim = self.get_dim('c_con')
        cg_dim = self.get_dim('c_gen')
        cv_dim = self.get_dim('c_var')
        hc_dim = self.get_dim('h_con')
        hg_dim = self.get_dim('h_gen')
        hv_dim = self.get_dim('h_var')
        as_dim = self.att_spec_dim

        if self.x_and_y_are_seqs:
            batch_size = x.shape[1]
        else:
            # if we're using "static" inputs, then we need to expand them out
            # into a proper sequential form
            batch_size = x.shape[0]
            x = x.dimshuffle('x',0,1).repeat(self.total_steps, axis=0)
            y = y.dimshuffle('x',0,1).repeat(self.total_steps, axis=0)

        # get initial states for all model components
        c0 = self.c_0.repeat(batch_size, axis=0)
        cc0 = self.cc_0.repeat(batch_size, axis=0)
        u_hc0 = 0.05 * self.theano_rng.normal( \
                        size=(batch_size, hc_dim), avg=0., std=1.)
        hc0 = self.hc_0.repeat(batch_size, axis=0) + u_hc0
        cg0 = self.cg_0.repeat(batch_size, axis=0)
        hg0 = self.hg_0.repeat(batch_size, axis=0)
        cv0 = self.cv_0.repeat(batch_size, axis=0)
        hv0 = self.hv_0.repeat(batch_size, axis=0)

        # get zero-mean, unit-std. Gaussian noise for use in scan op
        u = self.theano_rng.normal(
                    size=(self.total_steps, batch_size, z_dim),
                    avg=0., std=1.)

        u_att = self.att_noise * self.theano_rng.normal(
                        size=(self.total_steps, batch_size, as_dim),
                        avg=0., std=1.)

        # run the multi-stage guided generative process
        cs, h_cons, _, _, _, _, _, c_as_ys, nlls, kl_q2ps, kl_p2qs, att_maps, read_imgs = \
                self.iterate(x=x, y=y, u=u, u_att=u_att,
                             nll_scale=self.nll_scales,
                             c=c0,
                             h_con=hc0, c_con=cc0,
                             h_gen=hg0, c_gen=cg0,
                             h_var=hv0, c_var=cv0)

        # add name tags to the constructed values
        xs = x
        xs.name = "xs"
        cs.name = "cs"
        h_cons.name = "h_cons"
        c_as_ys.name = "c_as_ys"
        nlls.name = "nlls"
        kl_q2ps.name = "kl_q2ps"
        kl_p2qs.name = "kl_p2qs"
        att_maps.name = "att_maps"
        read_imgs.name = "read_imgs"
        return xs, cs, h_cons, c_as_ys, nlls, kl_q2ps, kl_p2qs, att_maps, read_imgs

    def build_model_funcs(self):
        """
        Build the symbolic costs and theano functions relevant to this model.
        """
        # some symbolic vars to represent various inputs/outputs
        if self.x_and_y_are_seqs:
            x_sym = tensor.tensor3('x_sym')
            y_sym = tensor.tensor3('y_sym')
        else:
            x_sym = tensor.matrix('x_sym')
            y_sym = tensor.matrix('y_sym')

        # collect estimates of y given x produced by this model
        xs, cs, h_cons, c_as_ys, nlls, kl_q2ps, kl_p2qs, att_maps, read_imgs = \
                self.process_inputs(x_sym, y_sym)

        # get the expected NLL part of the VFE bound
        self.nll_term = nlls.sum(axis=0).mean()
        self.nll_term.name = "nll_term"

        # get KL(q || p) and KL(p || q) and KL(p || g)
        self.kld_q2p_term = kl_q2ps.sum(axis=0).mean()
        self.kld_q2p_term.name = "kld_q2p_term"
        self.kld_p2q_term = kl_p2qs.sum(axis=0).mean()
        self.kld_p2q_term.name = "kld_p2q_term"

        # get the proper VFE bound on NLL
        self.nll_bound = self.nll_term + self.kld_q2p_term
        self.nll_bound.name = "nll_bound"

        # grab handles for all the optimizable parameters in our cost
        self.cg = ComputationGraph([self.nll_bound])
        self.joint_params = self.get_model_params(ary_type='theano')

        # apply some l2 regularization to the model parameters
        self.reg_term = (1e-5 * sum([tensor.sum(p**2.0) for p in self.joint_params]))
        self.reg_term.name = "reg_term"

        # compute the full cost w.r.t. which we will optimize params
        self.joint_cost = self.nll_term + \
                          (self.lam_kld_q2p[0] * self.kld_q2p_term) + \
                          (self.lam_kld_p2q[0] * self.kld_p2q_term) + \
                          self.reg_term
        self.joint_cost.name = "joint_cost"

        # get the gradient of the joint cost for all optimizable parameters
        print("Computing gradients of joint_cost...")
        self.joint_grads = OrderedDict()
        grad_list = tensor.grad(self.joint_cost, self.joint_params)
        for i, p in enumerate(self.joint_params):
            self.joint_grads[p] = grad_list[i]

        # construct the updates for all trainable parameters
        self.joint_updates = get_adam_updates(params=self.joint_params, \
                grads=self.joint_grads, alpha=self.lr, \
                beta1=self.mom_1, beta2=self.mom_2, \
                mom2_init=1e-3, smoothing=1e-5, max_grad_norm=10.0)

        # collect the outputs to return from this function
        outputs = [self.joint_cost, self.nll_term, \
                   self.kld_q2p_term, self.kld_p2q_term, self.reg_term]
        # collect the required inputs
        inputs = [x_sym, y_sym]

        # compile the theano functions for computing stuff, like for real
        print("Compiling model training/update function...")
        self.train_joint = theano.function(inputs=inputs, \
                                           outputs=outputs, \
                                           updates=self.joint_updates)
        print("Compiling model cost estimator function...")
        self.compute_nll_bound = theano.function(inputs=inputs, \
                                                 outputs=outputs)
        return

    def build_attention_funcs(self):
        """
        Build functions for visualizing the behavior of this model, assuming
        self.reader_mlp is SimpleAttentionReader2d or SimpleAttentionReader1d.
        """
        # some symbolic vars to represent various inputs/outputs
        if self.x_and_y_are_seqs:
            x_sym = tensor.tensor3('x_sym_att_funcs')
            y_sym = tensor.tensor3('y_sym_att_funcs')
        else:
            x_sym = tensor.matrix('x_sym_att_funcs')
            y_sym = tensor.matrix('y_sym_att_funcs')
        # collect estimates of y given x produced by this model
        xs, cs, h_cons, c_as_ys, nlls, kl_q2ps, kl_p2qs, att_maps, read_imgs = \
                self.process_inputs(x_sym, y_sym)
        # build the function for computing the attention trajectories
        print("Compiling attention tracker...")
        inputs = [x_sym, y_sym]
        outputs = [c_as_ys, att_maps, read_imgs, xs]
        sample_func = theano.function(inputs=inputs, \
                                      outputs=outputs)
        def switchy_sampler(x, y, sample_source='q'):
            # store value of sample source switch, to restore later
            old_switch = self.train_switch.get_value()
            if sample_source == 'p':
                # take samples from the primary policy
                zeros_ary = numpy.zeros((1,)).astype(theano.config.floatX)
                self.train_switch.set_value(zeros_ary)
            else:
                # take samples from the guide policy
                ones_ary = numpy.ones((1,)).astype(theano.config.floatX)
                self.train_switch.set_value(ones_ary)
            # sample prediction and attention trajectories
            outs = sample_func(x, y)
            xs = outs[-1]
            # set sample source switch back to previous value
            self.train_switch.set_value(old_switch)
            # grab prediction values
            y_preds = outs[0]
            # aggregate sequential attention maps
            obs_dim = self.x_dim
            seq_len = self.total_steps
            samp_count = outs[1].shape[1]
            att_count = 1
            a_maps = numpy.zeros((outs[1].shape[0], outs[1].shape[1], obs_dim))
            for a_num in range(att_count):
                start_idx = a_num * obs_dim
                end_idx = (a_num + 1) * obs_dim
                for s1 in range(samp_count):
                    for s2 in range(seq_len):
                        a_maps[s2,s1,:] = a_maps[s2,s1,:] + \
                                          outs[1][s2,s1,start_idx:end_idx]
            # aggregate sequential attention read-outs
            r_outs = numpy.zeros((outs[2].shape[0], outs[2].shape[1], obs_dim))
            for a_num in range(att_count):
                start_idx = a_num * obs_dim
                end_idx = (a_num + 1) * obs_dim
                for s1 in range(samp_count):
                    for s2 in range(seq_len):
                        r_outs[s2,s1,:] = r_outs[s2,s1,:] + \
                                          outs[2][s2,s1,start_idx:end_idx]
            result = [y_preds, a_maps, r_outs, xs]
            return result
        self.sample_attention = switchy_sampler
        return

    def get_model_params(self, ary_type='numpy'):
        """
        Get the optimizable parameters in this model. This returns a list
        and, to reload this model's parameters, the list must stay in order.

        This can provide shared variables or numpy arrays.
        """
        if self.cg is None:
            self.build_model_funcs()
        joint_params = VariableFilter(roles=[PARAMETER])(self.cg.variables)
        if ary_type == 'numpy':
            for i, p in enumerate(joint_params):
                joint_params[i] = p.get_value(borrow=False)
        return joint_params

    def set_model_params(self, numpy_param_list):
        """
        Set the optimizable parameters in this model. This requires a list
        and, to reload this model's parameters, the list must be in order.
        """
        if self.cg is None:
            self.build_model_funcs()
        # grab handles for all the optimizable parameters in our cost
        joint_params = VariableFilter(roles=[PARAMETER])(self.cg.variables)
        for i, p in enumerate(joint_params):
            joint_params[i].set_value(to_fX(numpy_param_list[i]))
        return joint_params

    def save_model_params(self, f_name=None):
        """
        Save model parameters to a pickle file, in numpy form.
        """
        numpy_params = self.get_model_params(ary_type='numpy')
        f_handle = file(f_name, 'wb')
        # dump the dict self.params, which just holds "simple" python values
        cPickle.dump(numpy_params, f_handle, protocol=-1)
        f_handle.close()
        return

    def load_model_params(self, f_name=None):
        """
        Load model parameters from a pickle file, in numpy form.
        """
        pickle_file = open(f_name)
        numpy_params = cPickle.load(pickle_file)
        self.set_model_params(numpy_params)
        pickle_file.close()
        return
