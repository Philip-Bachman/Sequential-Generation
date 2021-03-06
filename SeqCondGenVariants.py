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
from blocks.bricks import Random, MLP, Linear, Tanh, Softmax, Initializable
from blocks.bricks import Tanh, Identity, Activation, Feedforward
from blocks.bricks.cost import BinaryCrossEntropy
from blocks.utils import shared_floatx_nans
from blocks.roles import add_role, WEIGHT, BIAS, PARAMETER, AUXILIARY


from DKCode import get_adam_updates, get_adam_updates_X
from HelperFuncs import constFX, to_fX
from LogPDFs import log_prob_bernoulli, gaussian_kld, gaussian_ent, \
                    gaussian_logvar_kld, gaussian_mean_kld




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



class SeqCondGenALL(BaseRecurrent, Initializable, Random):
    """
    SeqCondGenALL -- constructs conditional densities under time constraints.

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
        x_dim: dimension of inputs on which to condition
        y_dim: dimension of outputs to predict
        use_var: whether to include "guide" distribution for observer
        use_rav: whether to include "guide" distribution for controller
        use_att: whether to use attention or read full inputs
        reader_mlp: used for reading from the input
        writer_mlp: used for writing to the output prediction
        con_mlp_in: preprocesses input to the "controller" LSTM
        con_rnn: the "controller" LSTM
        con_mlp_out: CondNet for distribution over att spec given con_rnn
        obs_mlp_in: preprocesses input to the "observer" LSTM
        obs_rnn: the "observer" LSTM
        obs_mlp_out: CondNet for distribution over z given gen_rnn
        var_mlp_in: preprocesses input to the "guide observer" LSTM
        var_rnn: the "guide observer" LSTM
        var_mlp_out: CondNet for distribution over z given var_rnn
        rav_mlp_in: preprocesses input to the "guide controller" LSTM
        rav_rnn: the "guide controller" LSTM
        rav_mlp_out: CondNet for distribution over z given rav_rnn
    """
    def __init__(self, x_and_y_are_seqs,
                    total_steps, init_steps,
                    exit_rate, nll_weight,
                    x_dim, y_dim,
                    use_var, use_rav, use_att,
                    reader_mlp, writer_mlp,
                    con_mlp_in, con_rnn, con_mlp_out,
                    obs_mlp_in, obs_rnn, obs_mlp_out,
                    var_mlp_in, var_rnn, var_mlp_out,
                    rav_mlp_in, rav_rnn, rav_mlp_out,
                    com_noise=0.1, att_noise=0.05,
                    **kwargs):
        super(SeqCondGenALL, self).__init__(**kwargs)
        # record basic structural parameters
        self.x_and_y_are_seqs = x_and_y_are_seqs
        self.total_steps = total_steps
        self.init_steps = init_steps
        self.exit_rate = exit_rate
        self.nll_weight = nll_weight
        self.use_var = use_var
        self.use_rav = use_rav
        self.use_att = use_att
        self.x_dim = x_dim
        self.y_dim = y_dim
        assert (self.x_dim == self.y_dim), "x_dim must equal y_dim!"
        # construct a sequence of scales for measuring NLL. we'll use scales
        # corresponding to some fixed number of guaranteed steps, followed by
        # a constant probability of early stopping. any "residual" probability
        # will be picked up by the final step.
        self.nll_scales = self._construct_nll_scales()
        # grab handles for shared read/write models
        self.reader_mlp = reader_mlp
        self.writer_mlp = writer_mlp
        # set up stuff for dealing with stochastic attention placement
        self.att_spec_dim = 5 # dimension for attention specification
        # grab handles for sequential read/write models
        self.con_mlp_in = con_mlp_in
        self.con_rnn = con_rnn
        self.con_mlp_out = con_mlp_out
        self.obs_mlp_in = obs_mlp_in
        self.obs_rnn = obs_rnn
        self.obs_mlp_out = obs_mlp_out
        self.var_mlp_in = var_mlp_in
        self.var_rnn = var_rnn
        self.var_mlp_out = var_mlp_out
        self.rav_mlp_in = rav_mlp_in
        self.rav_rnn = rav_rnn
        self.rav_mlp_out = rav_mlp_out
        # create a shared variable switch for controlling sampling
        ones_ary = numpy.ones((1,)).astype(theano.config.floatX)
        self.train_switch = theano.shared(value=ones_ary, name='train_switch')
        # create shared variables for controlling KLd terms
        self.lam_kld_q2p = theano.shared(value=ones_ary, name='lam_kld_q2p')
        self.lam_kld_p2q = theano.shared(value=ones_ary, name='lam_kld_p2q')
        self.lam_kld_amu = theano.shared(value=ones_ary, name='lam_kld_amu')
        self.lam_kld_alv = theano.shared(value=ones_ary, name='lam_kld_alv')
        self.set_lam_kld(lam_kld_q2p=1.0, lam_kld_p2q=0.1, \
                         lam_kld_amu=0.0, lam_kld_alv=0.1)
        # create shared variables for controlling optimization/updates
        self.lr = theano.shared(value=0.0001*ones_ary, name='lr')
        self.mom_1 = theano.shared(value=0.9*ones_ary, name='mom_1')
        self.mom_2 = theano.shared(value=0.99*ones_ary, name='mom_2')

        # set noise scale for the communication channel observer -> controller
        noise_ary = to_fX(com_noise * ones_ary)
        self.com_noise = theano.shared(value=noise_ary, name='com_noise')
        # set noise scale for the attention placement
        noise_ary = to_fX(att_noise * ones_ary)
        self.att_noise = theano.shared(value=noise_ary, name='att_noise')

        # setup a "null pointer" that will point to the computation graph
        # for this model, which can be built by self.build_model_funcs()...
        self.cg = None

        # record the sub-models around which this model is built
        self.params = []
        self.children = [self.reader_mlp, self.writer_mlp,
                         self.con_mlp_in, self.con_rnn, self.con_mlp_out,
                         self.obs_mlp_in, self.obs_rnn, self.obs_mlp_out,
                         self.var_mlp_in, self.var_rnn, self.var_mlp_out,
                         self.rav_mlp_in, self.rav_rnn, self.rav_mlp_out]
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

    def set_lam_kld(self, lam_kld_q2p=0.0, lam_kld_p2q=1.0, \
                    lam_kld_amu=0.0, lam_kld_alv=0.0):
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
        new_lam = zero_ary + lam_kld_amu
        self.lam_kld_amu.set_value(to_fX(new_lam))
        new_lam = zero_ary + lam_kld_alv
        self.lam_kld_alv.set_value(to_fX(new_lam))
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
        # self.c_0 provides initial state of the next column prediction
        self.c_0 = shared_floatx_nans((1,c_dim), name='c_0')
        add_role(self.c_0, PARAMETER)
        # self.cc_0/self.hc_0 provides initial state of the controller
        self.cc_0 = shared_floatx_nans((1,cc_dim), name='cc_0')
        add_role(self.cc_0, PARAMETER)
        self.hc_0 = shared_floatx_nans((1,hc_dim), name='hc_0')
        add_role(self.hc_0, PARAMETER)
        # self.co_0/self.ho_0 provides initial state of the primary policy
        self.co_0 = shared_floatx_nans((1,co_dim), name='co_0')
        add_role(self.co_0, PARAMETER)
        self.ho_0 = shared_floatx_nans((1,ho_dim), name='ho_0')
        add_role(self.ho_0, PARAMETER)
        # self.cv_0/self.hv_0 provides initial state of the guide policy
        self.cv_0 = shared_floatx_nans((1,cv_dim), name='cv_0')
        add_role(self.cv_0, PARAMETER)
        self.hv_0 = shared_floatx_nans((1,hv_dim), name='hv_0')
        add_role(self.hv_0, PARAMETER)
        # self.cr_0/self.hr_0 provides initial state of the guide policy
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
        elif name in ['c', 'c0', 'y']:
            return self.y_dim
        elif name in ['u_com', 'z']:
            return self.obs_mlp_out.get_dim('output')
        elif name in ['u_att']:
            return self.att_spec_dim
        elif name in ['h_con', 'hc0']:
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
        elif name in ['nll', 'kl_q2p', 'kl_p2q', 'kld_amu', 'kld_alv']:
            return 0
        else:
            super(SeqCondGenALL, self).get_dim(name)
        return

    #------------------------------------------------------------------------

    @recurrent(sequences=['x', 'y', 'u_com', 'u_att', 'nll_scale'], contexts=[],
               states=['c', 'h_con', 'c_con', 'h_obs', 'c_obs', 'h_var', 'c_var', 'h_rav', 'c_rav'],
               outputs=['c', 'h_con', 'c_con', 'h_obs', 'c_obs', 'h_var', 'c_var', 'h_rav', 'c_rav', 'c_as_y', 'nll', 'kl_q2p', 'kl_p2q', 'kl_amu', 'kl_alv', 'att_map', 'read_img'])
    def apply(self, x, y, u_com, u_att, nll_scale, c, h_con, c_con, h_obs, c_obs, h_var, c_var, h_rav, c_rav):
        # non-additive steps use c_con as a "latent workspace", which means
        # it needs to be transformed before being comparable to y.
        c_as_y = tensor.nnet.sigmoid(self.writer_mlp.apply(h_con))
        # compute difference between current prediction and target value
        y_d = y - c_as_y

        # estimate conditional over attention spec given h_con (from time t-1)
        p_a_mean, p_a_logvar, p_att_spec = \
                self.con_mlp_out.apply(h_con, u_att)

        # treat attention placement as a "deterministic action"
        q_a_mean, q_a_logvar, q_att_spec = p_a_mean, p_a_logvar, p_att_spec
        p_att_spec = p_a_mean
        q_att_spec = q_a_mean

        # compute KL(guide || primary) for attention control
        kl_q2p_a = tensor.sum(gaussian_kld(q_a_mean, q_a_logvar, \
                              p_a_mean, p_a_logvar), axis=1)
        kl_p2q_a = tensor.sum(gaussian_kld(p_a_mean, p_a_logvar, \
                              q_a_mean, q_a_logvar), axis=1)

        # mix samples from p/q based on value of self.train_switch
        _att_spec = (self.train_switch[0] * q_att_spec) + \
                    ((1.0 - self.train_switch[0]) * p_att_spec)
        att_spec = self.att_noise[0] * _att_spec

        if self.use_att:
            # apply the attention-based reader to the input in x
            read_out = self.reader_mlp.apply(x, x, att_spec)
            true_out = self.reader_mlp.apply(y, y, att_spec)
            att_map = self.reader_mlp.att_map(att_spec)
            read_img = self.reader_mlp.write(read_out, att_spec)
            # construct inputs to obs and var nets using attention
            obs_inp = tensor.concatenate([read_out, att_spec, h_con], axis=1)
            var_inp = tensor.concatenate([y, read_out, att_spec, h_con], axis=1)
        else:
            # dummy outputs for attentino visualization
            att_map = (0.0 * x) + 0.5
            read_img = x
            # construct inputs to obs and var nets using full observation
            obs_inp = tensor.concatenate([x, att_spec, h_con], axis=1)
            var_inp = tensor.concatenate([y, x, att_spec, h_con], axis=1)

        # update the primary observer RNN state
        i_obs = self.obs_mlp_in.apply(obs_inp)
        h_obs, c_obs = self.obs_rnn.apply(states=h_obs, cells=c_obs,
                                          inputs=i_obs, iterate=False)
        # estimate primary conditional over z given h_obs
        p_z_mean, p_z_logvar, p_z = \
                self.obs_mlp_out.apply(h_obs, u_com)

        if self.use_var:
            # update the guide observer RNN state
            i_var = self.var_mlp_in.apply(var_inp)
            h_var, c_var = self.var_rnn.apply(states=h_var, cells=c_var,
                                              inputs=i_var, iterate=False)
            # estimate guide conditional over z given h_var
            q_z_mean, q_z_logvar, q_z = \
                    self.var_mlp_out.apply(h_var, u_com)
        else:
            # use the observer -> controller channel as "deterministic action"
            q_z_mean, q_z_logvar, q_z = p_z_mean, p_z_logvar, p_z
            p_z = p_z_mean
            q_z = p_z_mean

        # compute KL(guide || primary) for channel: observer -> controller
        kl_q2p_z = tensor.sum(gaussian_kld(q_z_mean, q_z_logvar, \
                              p_z_mean, p_z_logvar), axis=1)
        kl_p2q_z = tensor.sum(gaussian_kld(p_z_mean, p_z_logvar, \
                              q_z_mean, q_z_logvar), axis=1)
        # mix samples from p/q based on value of self.train_switch
        _z = (self.train_switch[0] * q_z) + \
             ((1.0 - self.train_switch[0]) * p_z)
        z = self.com_noise[0] * _z

        # update the primary controller RNN state
        i_con = self.con_mlp_in.apply(tensor.concatenate([z], axis=1))
        h_con, c_con = self.con_rnn.apply(states=h_con, cells=c_con, \
                                          inputs=i_con, iterate=False)

        # update the "workspace" (stored in c)
        c = self.writer_mlp.apply(h_con)

        # compute the NLL of the reconstruction as of this step. the NLL at
        # each step is rescaled by a factor such that the sum of the factors
        # for all steps is 1, and all factors are non-negative.
        c_as_y = tensor.nnet.sigmoid(c)
        nll = -nll_scale * tensor.flatten(log_prob_bernoulli(y, c_as_y))
        # compute KL(q || p) and KL(p || q) for this step
        kl_q2p = kl_q2p_z + kl_q2p_a
        kl_p2q = kl_p2q_z + kl_p2q_a
        # compute attention module KLd terms for this step
        kl_amu = tensor.sum(gaussian_mean_kld(p_a_mean, p_a_logvar, \
                            0., 0.), axis=1)
        kl_alv = tensor.sum(gaussian_logvar_kld(p_a_mean, p_a_logvar, \
                            0., 0.), axis=1)
        return c, h_con, c_con, h_obs, c_obs, h_var, c_var, h_rav, c_rav, c_as_y, nll, kl_q2p, kl_p2q, kl_amu, kl_alv, att_map, read_img

    #------------------------------------------------------------------------

    @application(inputs=['x', 'y'],
                 outputs=['xs', 'cs', 'c_as_ys', 'nlls', 'kl_q2ps', 'kl_p2qs', 'kl_amus', 'kl_alvs', 'att_maps', 'read_imgs'])
    def process_inputs(self, x, y):
        # get important size and shape information
        z_dim = self.get_dim('z')
        cc_dim = self.get_dim('c_con')
        hc_dim = self.get_dim('h_con')
        co_dim = self.get_dim('c_obs')
        ho_dim = self.get_dim('h_obs')
        cv_dim = self.get_dim('c_var')
        hv_dim = self.get_dim('h_var')
        cr_dim = self.get_dim('c_rav')
        hr_dim = self.get_dim('h_rav')
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
        hc0 = self.hc_0.repeat(batch_size, axis=0)
        co0 = self.co_0.repeat(batch_size, axis=0)
        ho0 = self.ho_0.repeat(batch_size, axis=0)
        cv0 = self.cv_0.repeat(batch_size, axis=0)
        hv0 = self.hv_0.repeat(batch_size, axis=0)
        cr0 = self.cr_0.repeat(batch_size, axis=0)
        hr0 = self.hr_0.repeat(batch_size, axis=0)

        # get zero-mean, unit-std. Gaussian noise for use in scan op
        u_com = self.theano_rng.normal(
                        size=(self.total_steps, batch_size, z_dim),
                        avg=0., std=1.)
        u_att = self.theano_rng.normal(
                        size=(self.total_steps, batch_size, as_dim),
                        avg=0., std=1.)

        # run the multi-stage guided generative process
        cs, _, _, _, _, _, _, _, _, c_as_ys, nlls, kl_q2ps, kl_p2qs, kl_amus, kl_alvs, att_maps, read_imgs = \
                self.apply(x=x, y=y, u_com=u_com, u_att=u_att,
                             nll_scale=self.nll_scales,
                             c=c0,
                             h_con=hc0, c_con=cc0,
                             h_obs=ho0, c_obs=co0,
                             h_var=hv0, c_var=cv0,
                             h_rav=hr0, c_rav=cr0)

        # add name tags to the constructed values
        xs = x
        xs.name = "xs"
        cs.name = "cs"
        c_as_ys.name = "c_as_ys"
        nlls.name = "nlls"
        kl_q2ps.name = "kl_q2ps"
        kl_p2qs.name = "kl_p2qs"
        kl_amus.name = "kl_amus"
        kl_alvs.name = "kl_alvs"
        att_maps.name = "att_maps"
        read_imgs.name = "read_imgs"
        return xs, cs, c_as_ys, nlls, kl_q2ps, kl_p2qs, kl_amus, kl_alvs, att_maps, read_imgs

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
        xs, cs, c_as_ys, nlls, kl_q2ps, kl_p2qs, kl_amus, kl_alvs, att_maps, read_imgs = \
                self.process_inputs(x_sym, y_sym)

        # get the expected NLL part of the VFE bound
        self.nll_term = nlls.sum(axis=0).mean()
        self.nll_term.name = "nll_term"

        # get KL(q || p) and KL(p || q) and KL(p || g)
        self.kld_q2p_term = kl_q2ps.sum(axis=0).mean()
        self.kld_q2p_term.name = "kld_q2p_term"
        self.kld_p2q_term = kl_p2qs.sum(axis=0).mean()
        self.kld_p2q_term.name = "kld_p2q_term"
        # get KLd terms on attention placement
        self.kld_amu_term = kl_amus.sum(axis=0).mean()
        self.kld_amu_term.name = "kld_amu_term"
        self.kld_alv_term = kl_alvs.sum(axis=0).mean()
        self.kld_alv_term.name = "kld_alv_term"

        # grab handles for all the optimizable parameters in our cost
        dummy_cost = self.nll_term + self.kld_q2p_term + self.kld_amu_term
        self.cg = ComputationGraph([dummy_cost])
        self.joint_params = self.get_model_params(ary_type='theano')

        # apply some l2 regularization to the model parameters
        self.reg_term = (1e-5 * sum([tensor.sum(p**2.0) for p in self.joint_params]))
        self.reg_term.name = "reg_term"

        # compute the full cost w.r.t. which we will optimize params
        self.joint_cost = self.nll_term + \
                          (self.lam_kld_q2p[0] * self.kld_q2p_term) + \
                          (self.lam_kld_p2q[0] * self.kld_p2q_term) + \
                          (self.lam_kld_amu[0] * self.kld_amu_term) + \
                          (self.lam_kld_alv[0] * self.kld_alv_term) + \
                          self.reg_term
        self.joint_cost.name = "joint_cost"

        # get the gradient of the joint cost for all optimizable parameters
        print("Computing gradients of joint_cost...")
        self.joint_grads = OrderedDict()
        grad_list = tensor.grad(self.joint_cost, self.joint_params)
        for i, p in enumerate(self.joint_params):
            self.joint_grads[p] = grad_list[i]

        # construct the updates for all trainable parameters
        self.joint_updates, applied_updates = get_adam_updates_X( \
                params=self.joint_params, \
                grads=self.joint_grads, alpha=self.lr, \
                beta1=self.mom_1, beta2=self.mom_2, \
                mom2_init=1e-3, smoothing=1e-4, max_grad_norm=10.0)

        # get the total grad norm and (post ADAM scaling) update norm.
        self.grad_norm = sum([tensor.sum(g**2.0) for g in grad_list])
        self.update_norm = sum([tensor.sum(u**2.0) for u in applied_updates])

        # collect the outputs to return from this function
        train_outputs = [self.joint_cost, self.nll_term, \
                   self.kld_q2p_term, self.kld_p2q_term, \
                   self.kld_amu_term, self.kld_alv_term, \
                   self.reg_term, self.grad_norm, self.update_norm]
        bound_outputs = [self.joint_cost, self.nll_term, \
                   self.kld_q2p_term, self.kld_p2q_term, \
                   self.kld_amu_term, self.kld_alv_term, \
                   self.reg_term]
        # collect the required inputs
        inputs = [x_sym, y_sym]

        # compile the theano functions for computing stuff, like for real
        print("Compiling model training/update function...")
        self.train_joint = theano.function(inputs=inputs, \
                                           outputs=train_outputs, \
                                           updates=self.joint_updates)
        print("Compiling model cost estimator function...")
        self.compute_nll_bound = theano.function(inputs=inputs, \
                                                 outputs=bound_outputs)
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
        xs, cs, c_as_ys, nlls, kl_q2ps, kl_p2qs, kl_amus, kl_alvs, att_maps, read_imgs = \
                self.process_inputs(x_sym, y_sym)

        # get the expected NLL part of the VFE bound
        self.nll_term = nlls.sum(axis=0).mean()
        self.nll_term.name = "nll_term"
        # get KL(q || p) and KL(p || q) and KL(p || g)
        self.kld_q2p_term = kl_q2ps.sum(axis=0).mean()
        self.kld_q2p_term.name = "kld_q2p_term"
        self.kld_p2q_term = kl_p2qs.sum(axis=0).mean()
        self.kld_p2q_term.name = "kld_p2q_term"
        # get KLd terms on attention placement
        self.kld_amu_term = kl_amus.sum(axis=0).mean()
        self.kld_amu_term.name = "kld_amu_term"
        self.kld_alv_term = kl_alvs.sum(axis=0).mean()
        self.kld_alv_term.name = "kld_alv_term"
        # grab handles for all the optimizable parameters in our cost
        dummy_cost = self.nll_term + self.kld_q2p_term + self.kld_amu_term
        self.cg = ComputationGraph([dummy_cost])

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
            map_count = int(outs[1].shape[2] / obs_dim)
            a_maps = numpy.zeros((outs[1].shape[0], outs[1].shape[1], obs_dim))
            for m_num in range(map_count):
                start_idx = m_num * obs_dim
                end_idx = (m_num + 1) * obs_dim
                for s1 in range(samp_count):
                    for s2 in range(seq_len):
                        a_maps[s2,s1,:] = a_maps[s2,s1,:] + \
                                          outs[1][s2,s1,start_idx:end_idx]
            # aggregate sequential attention read-outs
            r_outs = numpy.zeros((outs[2].shape[0], outs[2].shape[1], obs_dim))
            for m_num in range(map_count):
                start_idx = m_num * obs_dim
                end_idx = (m_num + 1) * obs_dim
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






################################
################################
## ATTENTION-BASED PERCEPTION ##
################################
################################

class SeqCondGenALT(BaseRecurrent, Initializable, Random):
    """
    SeqCondGenALT -- constructs conditional densities under time constraints.

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
        x_dim: dimension of inputs on which to condition
        y_dim: dimension of outputs to predict
        use_var: whether to include "guide" distribution for observer
        use_att: whether to use attention or read full inputs
        reader_mlp: used for reading from the input
        writer_mlp: transform shared dynamics state into a prediction for y
        pol1_mlp_in:
        pol1_rnn:
        pol2_mlp_in:
        pol2_rnn:
        pol_mlp_out: conditional over z given h_pol2
        var1_mlp_in:
        var1_rnn:
        var2_mlp_in:
        var2_rnn:
        var_mlp_out: conditional over z given h_var2
        dyn_mlp_in:
        dyn_rnn:
        att_spec_mlp: convert z into an attention spec
    """
    def __init__(self, x_and_y_are_seqs,
                    total_steps, init_steps,
                    exit_rate, nll_weight,
                    x_dim, y_dim,
                    use_var, use_att,
                    reader_mlp, writer_mlp,
                    pol1_mlp_in, pol1_rnn,
                    pol2_mlp_in, pol2_rnn,
                    pol_mlp_out,
                    var1_mlp_in, var1_rnn,
                    var2_mlp_in, var2_rnn,
                    var_mlp_out,
                    dyn_mlp_in, dyn_rnn,
                    att_spec_mlp,
                    **kwargs):
        super(SeqCondGenALT, self).__init__(**kwargs)
        # record basic structural parameters
        self.x_and_y_are_seqs = x_and_y_are_seqs
        self.total_steps = total_steps
        self.init_steps = init_steps
        self.exit_rate = exit_rate
        self.nll_weight = nll_weight
        self.use_var = use_var
        self.use_att = use_att
        self.x_dim = x_dim
        self.y_dim = y_dim
        assert (self.x_dim == self.y_dim), "x_dim must equal y_dim!"
        # construct a sequence of scales for measuring NLL. we'll use scales
        # corresponding to some fixed number of guaranteed steps, followed by
        # a constant probability of early stopping. any "residual" probability
        # will be picked up by the final step.
        self.nll_scales = self._construct_nll_scales()
        # grab handles for shared read/write models
        self.reader_mlp = reader_mlp
        self.writer_mlp = writer_mlp
        # set up stuff for dealing with stochastic attention placement
        self.att_spec_dim = 5 # dimension for attention specification
        # grab handles for sequential read/write models
        self.pol1_mlp_in = pol1_mlp_in
        self.pol1_rnn = pol1_rnn
        self.pol2_mlp_in = pol2_mlp_in
        self.pol2_rnn = pol2_rnn
        self.pol_mlp_out = pol_mlp_out
        self.var1_mlp_in = var1_mlp_in
        self.var1_rnn = var1_rnn
        self.var2_mlp_in = var2_mlp_in
        self.var2_rnn = var2_rnn
        self.var_mlp_out = var_mlp_out
        self.dyn_mlp_in = dyn_mlp_in
        self.dyn_rnn = dyn_rnn
        self.att_spec_mlp = att_spec_mlp
        # create a shared variable switch for controlling sampling
        ones_ary = numpy.ones((1,)).astype(theano.config.floatX)
        self.train_switch = theano.shared(value=ones_ary, name='train_switch')
        # create shared variables for controlling KLd terms
        self.lam_kld_q2p = theano.shared(value=ones_ary, name='lam_kld_q2p')
        self.lam_kld_p2q = theano.shared(value=ones_ary, name='lam_kld_p2q')
        self.lam_kld_amu = theano.shared(value=ones_ary, name='lam_kld_amu')
        self.lam_kld_alv = theano.shared(value=ones_ary, name='lam_kld_alv')
        self.set_lam_kld(lam_kld_q2p=1.0, lam_kld_p2q=0.1, \
                         lam_kld_amu=0.0, lam_kld_alv=0.1)
        # create shared variables for controlling optimization/updates
        self.lr = theano.shared(value=0.0001*ones_ary, name='lr')
        self.mom_1 = theano.shared(value=0.9*ones_ary, name='mom_1')
        self.mom_2 = theano.shared(value=0.99*ones_ary, name='mom_2')

        # setup a "null pointer" that will point to the computation graph
        # for this model, which can be built by self.build_model_funcs()...
        self.cg = None

        # record the sub-models around which this model is built
        self.params = []
        self.children = [self.reader_mlp, self.writer_mlp,
                         self.pol1_mlp_in, self.pol1_rnn,
                         self.pol2_mlp_in, self.pol2_rnn,
                         self.pol_mlp_out,
                         self.var1_mlp_in, self.var1_rnn,
                         self.var2_mlp_in, self.var2_rnn,
                         self.var_mlp_out,
                         self.dyn_mlp_in, self.dyn_rnn,
                         self.att_spec_mlp]
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

    def set_lam_kld(self, lam_kld_q2p=0.0, lam_kld_p2q=1.0, \
                    lam_kld_amu=0.0, lam_kld_alv=0.0):
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
        new_lam = zero_ary + lam_kld_amu
        self.lam_kld_amu.set_value(to_fX(new_lam))
        new_lam = zero_ary + lam_kld_alv
        self.lam_kld_alv.set_value(to_fX(new_lam))
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
        # initial attention spec
        z_dim = self.get_dim('z')
        self.z_0 = shared_floatx_nans((1,z_dim), name='z_0')

        # initial state of primary policy LSTM (1)
        hp1_dim = self.get_dim('h_pol1')
        cp1_dim = self.get_dim('c_pol1')
        self.hp1_0 = shared_floatx_nans((1,hp1_dim), name='hp1_0')
        self.cp1_0 = shared_floatx_nans((1,cp1_dim), name='cp1_0')
        add_role(self.hp1_0, PARAMETER)
        add_role(self.cp1_0, PARAMETER)
        # initial state of primary policy LSTM (2)
        hp2_dim = self.get_dim('h_pol2')
        cp2_dim = self.get_dim('c_pol2')
        self.hp2_0 = shared_floatx_nans((1,hp2_dim), name='hp2_0')
        self.cp2_0 = shared_floatx_nans((1,cp2_dim), name='cp2_0')
        add_role(self.hp2_0, PARAMETER)
        add_role(self.cp2_0, PARAMETER)

        # initial state of guide policy LSTM (1)
        hv1_dim = self.get_dim('h_var1')
        cv1_dim = self.get_dim('c_var1')
        self.hv1_0 = shared_floatx_nans((1,hv1_dim), name='hv1_0')
        self.cv1_0 = shared_floatx_nans((1,cv1_dim), name='cv1_0')
        add_role(self.hv1_0, PARAMETER)
        add_role(self.cv1_0, PARAMETER)
        # initial state of guide policy LSTM (2)
        hv2_dim = self.get_dim('h_var2')
        cv2_dim = self.get_dim('c_var2')
        self.hv2_0 = shared_floatx_nans((1,hv2_dim), name='hv2_0')
        self.cv2_0 = shared_floatx_nans((1,cv2_dim), name='cv2_0')
        add_role(self.hv2_0, PARAMETER)
        add_role(self.cv2_0, PARAMETER)

        # initial state of shared dynamics LSTM
        hd_dim = self.get_dim('h_dyn')
        cd_dim = self.get_dim('c_dyn')
        self.hd_0 = shared_floatx_nans((1,hd_dim), name='hd_0')
        self.cd_0 = shared_floatx_nans((1,cd_dim), name='cd_0')
        add_role(self.hd_0, PARAMETER)
        add_role(self.cd_0, PARAMETER)

        # add the theano shared variables to our parameter lists
        self.params.extend([ self.z_0, \
                             self.hp1_0, self.hp2_0, self.hv1_0, self.hv2_0,
                             self.cp1_0, self.cp2_0, self.cv1_0, self.cv2_0,
                             self.hd_0, self.cd_0 ])
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
        elif name == 'y':
            return self.y_dim
        elif name == 'nll_scale':
            return 1
        elif name == 'att_spec':
            return self.att_spec_dim
        elif name in ['u', 'z']:
            return self.pol_mlp_out.get_dim('output')
        elif name in ['h_pol1', 'hp10']:
            return self.pol1_rnn.get_dim('states')
        elif name in ['h_pol2', 'hp20']:
            return self.pol2_rnn.get_dim('states')
        elif name in ['h_var1', 'hv10']:
            return self.var1_rnn.get_dim('states')
        elif name in ['h_var2', 'hv20']:
            return self.var2_rnn.get_dim('states')
        elif name in ['h_dyn', 'hd0']:
            return self.dyn_rnn.get_dim('states')
        elif name in ['c_pol1', 'cp10']:
            return self.pol1_rnn.get_dim('cells')
        elif name in ['c_pol2', 'cp20']:
            return self.pol2_rnn.get_dim('cells')
        elif name in ['c_var1', 'cv10']:
            return self.var1_rnn.get_dim('cells')
        elif name in ['c_var2', 'cv20']:
            return self.var2_rnn.get_dim('cells')
        elif name in ['c_dyn', 'cd0']:
            return self.dyn_rnn.get_dim('cells')
        elif name in ['nll', 'kl_q2p', 'kl_p2q', 'kld_amu', 'kld_alv']:
            return 0
        else:
            super(SeqCondGenALT, self).get_dim(name)
        return

    #------------------------------------------------------------------------

    @recurrent(sequences=['x', 'y', 'u', 'nll_scale'], contexts=[],
               states=['z', 'h_pol1', 'c_pol1', 'h_pol2', 'c_pol2', 'h_var1', 'c_var1', 'h_var2', 'c_var2', 'h_dyn', 'c_dyn'],
               outputs=['z', 'h_pol1', 'c_pol1', 'h_pol2', 'c_pol2', 'h_var1', 'c_var1', 'h_var2', 'c_var2', 'h_dyn', 'c_dyn', 'y_pred', 'nll', 'kl_q2p', 'kl_p2q', 'att_map', 'read_img'])
    def apply(self, x, y, u, nll_scale, z, h_pol1, c_pol1, h_pol2, c_pol2, h_var1, c_var1, h_var2, c_var2, h_dyn, c_dyn):
        # decode attention placement from state of shared dynamics
        att_spec = self.att_spec_mlp.apply(h_dyn)
        if self.use_att:
            # apply the attention-based reader to the input in x
            read_out = self.reader_mlp.apply(x, x, att_spec)
            true_out = self.reader_mlp.apply(y, y, att_spec)
            att_map = self.reader_mlp.att_map(att_spec)
            read_img = self.reader_mlp.write(read_out, att_spec)
            # construct inputs to pol and var nets using attention
            pol_inp = tensor.concatenate([read_out, att_spec, h_dyn], axis=1)
            var_inp = tensor.concatenate([y, read_out, att_spec, h_dyn], axis=1)
        else:
            # dummy outputs for attention visualization
            att_map = (0.0 * x) + 0.5
            read_img = x
            # construct inputs to obs and var nets using full observation
            pol_inp = tensor.concatenate([x, att_spec, h_dyn], axis=1)
            var_inp = tensor.concatenate([y, x, att_spec, h_dyn], axis=1)

        # update the primary policy's deep LSTM state
        i_pol1 = self.pol1_mlp_in.apply(pol_inp)
        h_pol1, c_pol1 = self.pol1_rnn.apply(states=h_pol1, cells=c_pol1,
                                             inputs=i_pol1, iterate=False)
        i_pol2 = self.pol2_mlp_in.apply(h_pol1)
        h_pol2, c_pol2 = self.pol2_rnn.apply(states=h_pol2, cells=c_pol2,
                                             inputs=i_pol2, iterate=False)
        # estimate primary policy's conditional over z given h_pol2
        p_z_mean, p_z_logvar, p_z = \
                self.pol_mlp_out.apply(h_pol2, u)

        if self.use_var:
            # update the guide policy's deep LSTM state
            i_var1 = self.var1_mlp_in.apply(var_inp)
            h_var1, c_var1 = self.var1_rnn.apply(states=h_var1, cells=c_var1,
                                                 inputs=i_var1, iterate=False)
            i_var2 = self.var2_mlp_in.apply(h_var1)
            h_var2, c_var2 = self.var2_rnn.apply(states=h_var2, cells=c_var2,
                                                 inputs=i_var2, iterate=False)
            # estimate guide policy's conditional over z given h_var2
            q_z_mean, q_z_logvar, q_z = \
                    self.var_mlp_out.apply(h_var2, u)
        else:
            # use the observer -> controller channel as "deterministic action"
            q_z_mean, q_z_logvar, q_z = p_z_mean, p_z_logvar, p_z
            p_z = p_z_mean
            q_z = p_z_mean

        # mix samples from p/q based on value of self.train_switch
        z = (self.train_switch[0] * q_z) + \
             ((1.0 - self.train_switch[0]) * p_z)
        # compute KL(guide || primary) and KL(primary || guide)
        kl_q2p = tensor.sum(gaussian_kld(q_z_mean, q_z_logvar, \
                            p_z_mean, p_z_logvar), axis=1)
        kl_p2q = tensor.sum(gaussian_kld(p_z_mean, p_z_logvar, \
                            q_z_mean, q_z_logvar), axis=1)

        # update the shared dynamics LSTM state
        inp_dyn = tensor.concatenate([z], axis=1)
        i_dyn = self.dyn_mlp_in.apply(inp_dyn)
        h_dyn, c_dyn = self.dyn_rnn.apply(states=h_dyn, cells=c_dyn, \
                                          inputs=i_dyn, iterate=False)

        # transform the dynamics state into an observation/prediction
        y_pred = tensor.nnet.sigmoid(self.writer_mlp.apply(h_dyn))

        # compute the NLL of the reconstruction as of this step. the NLL at
        # each step is rescaled by a factor such that the sum of the factors
        # for all steps is 1, and all factors are non-negative.
        nll = -nll_scale * tensor.flatten(log_prob_bernoulli(y, y_pred))
        return z, h_pol1, c_pol1, h_pol2, c_pol2, h_var1, c_var1, h_var2, c_var2, h_dyn, c_dyn, y_pred, nll, kl_q2p, kl_p2q, att_map, read_img

    #------------------------------------------------------------------------

    @application(inputs=['x', 'y'],
                 outputs=['xs', 'y_preds', 'nlls', 'kl_q2ps', 'kl_p2qs', 'att_maps', 'read_imgs'])
    def process_inputs(self, x, y):
        # get important size and shape information
        z_dim = self.get_dim('z')
        hp1_dim = self.get_dim('h_pol1')
        cp1_dim = self.get_dim('c_pol1')
        hp2_dim = self.get_dim('h_pol2')
        cp2_dim = self.get_dim('c_pol2')
        hv1_dim = self.get_dim('h_var1')
        cv1_dim = self.get_dim('c_var1')
        hv2_dim = self.get_dim('h_var2')
        cv2_dim = self.get_dim('c_var2')
        hd_dim = self.get_dim('h_dyn')
        cd_dim = self.get_dim('c_dyn')

        if self.x_and_y_are_seqs:
            batch_size = x.shape[1]
        else:
            # if we're using "static" inputs, then we need to expand them out
            # into a proper sequential form
            batch_size = x.shape[0]
            x = x.dimshuffle('x',0,1).repeat(self.total_steps, axis=0)
            y = y.dimshuffle('x',0,1).repeat(self.total_steps, axis=0)

        # get initial states for all model components
        z0 = self.z_0.repeat(batch_size, axis=0)
        hp10 = self.hp1_0.repeat(batch_size, axis=0)
        cp10 = self.cp1_0.repeat(batch_size, axis=0)
        hp20 = self.hp2_0.repeat(batch_size, axis=0)
        cp20 = self.cp2_0.repeat(batch_size, axis=0)
        hv10 = self.hv1_0.repeat(batch_size, axis=0)
        cv10 = self.cv1_0.repeat(batch_size, axis=0)
        hv20 = self.hv2_0.repeat(batch_size, axis=0)
        cv20 = self.cv2_0.repeat(batch_size, axis=0)
        hd0 = self.hd_0.repeat(batch_size, axis=0)
        cd0 = self.cd_0.repeat(batch_size, axis=0)

        # get zero-mean, unit-std. Gaussian noise for use in scan op
        u = self.theano_rng.normal(
                    size=(self.total_steps, batch_size, z_dim),
                    avg=0., std=1.)

        # run the multi-stage guided generative process
        _, _, _, _, _, _, _, _, _, _, _, y_preds, nlls, kl_q2ps, kl_p2qs, att_maps, read_imgs = \
                self.apply(x=x, y=y, u=u, nll_scale=self.nll_scales,
                           z=z0,
                           h_pol1=hp10, c_pol1=cp10,
                           h_pol2=hp20, c_pol2=cp20,
                           h_var1=hv10, c_var1=cv10,
                           h_var2=hv20, c_var2=cv20,
                           h_dyn=hd0, c_dyn=cd0)

        # add name tags to the constructed values
        xs = x
        xs.name = "xs"
        y_preds.name = "y_preds"
        nlls.name = "nlls"
        kl_q2ps.name = "kl_q2ps"
        kl_p2qs.name = "kl_p2qs"
        att_maps.name = "att_maps"
        read_imgs.name = "read_imgs"
        return xs, y_preds, nlls, kl_q2ps, kl_p2qs, att_maps, read_imgs

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
        xs, y_preds, nlls, kl_q2ps, kl_p2qs, att_maps, read_imgs = \
                self.process_inputs(x_sym, y_sym)

        # get the expected NLL part of the VFE bound
        self.nll_term = nlls.sum(axis=0).mean()
        self.nll_term.name = "nll_term"

        # get KL(q || p) and KL(p || q) and KL(p || g)
        self.kld_q2p_term = kl_q2ps.sum(axis=0).mean()
        self.kld_q2p_term.name = "kld_q2p_term"
        self.kld_p2q_term = kl_p2qs.sum(axis=0).mean()
        self.kld_p2q_term.name = "kld_p2q_term"

        # grab handles for all the optimizable parameters in our cost
        dummy_cost = self.nll_term + self.kld_q2p_term
        self.cg = ComputationGraph([dummy_cost])
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
        self.joint_updates, applied_updates = get_adam_updates_X( \
                params=self.joint_params, \
                grads=self.joint_grads, alpha=self.lr, \
                beta1=self.mom_1, beta2=self.mom_2, \
                mom2_init=1e-3, smoothing=1e-4, max_grad_norm=10.0)

        # get the total grad norm and (post ADAM scaling) update norm.
        self.grad_norm = sum([tensor.sum(g**2.0) for g in grad_list])
        self.update_norm = sum([tensor.sum(u**2.0) for u in applied_updates])

        # collect the outputs to return from this function
        train_outputs = [self.joint_cost, self.nll_term, \
                   self.kld_q2p_term, self.kld_p2q_term, \
                   self.reg_term, self.grad_norm, self.update_norm]
        bound_outputs = [self.joint_cost, self.nll_term, \
                   self.kld_q2p_term, self.kld_p2q_term, \
                   self.reg_term]
        # collect the required inputs
        inputs = [x_sym, y_sym]

        # compile the theano functions for computing stuff, like for real
        print("Compiling model training/update function...")
        self.train_joint = theano.function(inputs=inputs, \
                                           outputs=train_outputs, \
                                           updates=self.joint_updates)
        print("Compiling model cost estimator function...")
        self.compute_nll_bound = theano.function(inputs=inputs, \
                                                 outputs=bound_outputs)
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
        xs, y_preds, nlls, kl_q2ps, kl_p2qs, att_maps, read_imgs = \
                self.process_inputs(x_sym, y_sym)

        # get the expected NLL part of the VFE bound
        self.nll_term = nlls.sum(axis=0).mean()
        self.nll_term.name = "nll_term"
        # get KL(q || p) and KL(p || q) and KL(p || g)
        self.kld_q2p_term = kl_q2ps.sum(axis=0).mean()
        self.kld_q2p_term.name = "kld_q2p_term"
        self.kld_p2q_term = kl_p2qs.sum(axis=0).mean()
        self.kld_p2q_term.name = "kld_p2q_term"
        # grab handles for all the optimizable parameters in our cost
        dummy_cost = self.nll_term + self.kld_q2p_term
        self.cg = ComputationGraph([dummy_cost])

        # build the function for computing the attention trajectories
        print("Compiling attention tracker...")
        inputs = [x_sym, y_sym]
        outputs = [y_preds, att_maps, read_imgs, xs]
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
            ypreds = outs[0]
            # aggregate sequential attention maps
            obs_dim = self.x_dim
            seq_len = self.total_steps
            samp_count = outs[1].shape[1]
            map_count = int(outs[1].shape[2] / obs_dim)
            a_maps = numpy.zeros((outs[1].shape[0], outs[1].shape[1], obs_dim))
            for m_num in range(map_count):
                start_idx = m_num * obs_dim
                end_idx = (m_num + 1) * obs_dim
                for s1 in range(samp_count):
                    for s2 in range(seq_len):
                        a_maps[s2,s1,:] = a_maps[s2,s1,:] + \
                                          outs[1][s2,s1,start_idx:end_idx]
            # aggregate sequential attention read-outs
            r_outs = numpy.zeros((outs[2].shape[0], outs[2].shape[1], obs_dim))
            for m_num in range(map_count):
                start_idx = m_num * obs_dim
                end_idx = (m_num + 1) * obs_dim
                for s1 in range(samp_count):
                    for s2 in range(seq_len):
                        r_outs[s2,s1,:] = r_outs[s2,s1,:] + \
                                          outs[2][s2,s1,start_idx:end_idx]
            result = [ypreds, a_maps, r_outs, xs]
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

################################
################################
## ATTENTION-BASED PERCEPTION ##
################################
################################

class SeqCondGenALU(BaseRecurrent, Initializable, Random):
    """
    SeqCondGenALU -- constructs conditional densities under time constraints.

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
        x_dim: dimension of inputs on which to condition
        y_dim: dimension of outputs to predict
        use_var: whether to include "guide" distribution for observer
        use_att: whether to use attention or read full inputs
        reader_mlp: used for reading from the input
        pol1_mlp_in:
        pol1_rnn:
        pol1_mlp_out:
        pol2_mlp_in:
        pol2_rnn:
        pol2_mlp_out:
        var1_mlp_in:
        var1_rnn:
        var1_mlp_out:
        var2_mlp_in:
        var2_rnn: second LSTM for guide policy
        var2_mlp_out: estimate p(z | h_var2)
        dec_mlp_attn: convert z into an attention spec
        dec_mlp_self: convert z into feedback information
        dec_mlp_pred: convert z into a prediction for y
    """
    def __init__(self, x_and_y_are_seqs,
                    total_steps, init_steps,
                    exit_rate, nll_weight,
                    x_dim, y_dim,
                    use_var, use_att,
                    reader_mlp,
                    pol1_mlp_in, pol1_rnn, pol1_mlp_out,
                    pol2_mlp_in, pol2_rnn, pol2_mlp_out,
                    var1_mlp_in, var1_rnn, var1_mlp_out,
                    var2_mlp_in, var2_rnn, var2_mlp_out,
                    dec_mlp_attn, dec_mlp_self, dec_mlp_pred,
                    **kwargs):
        super(SeqCondGenALU, self).__init__(**kwargs)
        # record basic structural parameters
        self.x_and_y_are_seqs = x_and_y_are_seqs
        self.total_steps = total_steps
        self.init_steps = init_steps
        self.exit_rate = exit_rate
        self.nll_weight = nll_weight
        self.use_var = use_var
        self.use_att = use_att
        self.x_dim = x_dim
        self.y_dim = y_dim
        assert (self.x_dim == self.y_dim), "x_dim must equal y_dim!"
        # construct a sequence of scales for measuring NLL. we'll use scales
        # corresponding to some fixed number of guaranteed steps, followed by
        # a constant probability of early stopping. any "residual" probability
        # will be picked up by the final step.
        self.nll_scales = self._construct_nll_scales()
        # grab handle for shared attention-based reader
        self.reader_mlp = reader_mlp
        # set up stuff for dealing with stochastic attention placement
        self.att_spec_dim = 5 # dimension for attention specification
        # grab handles for sequential read/write models
        self.pol1_mlp_in = pol1_mlp_in
        self.pol1_rnn = pol1_rnn
        self.pol1_mlp_out = pol1_mlp_out
        self.pol2_mlp_in = pol2_mlp_in
        self.pol2_rnn = pol2_rnn
        self.pol2_mlp_out = pol2_mlp_out
        self.var1_mlp_in = var1_mlp_in
        self.var1_rnn = var1_rnn
        self.var1_mlp_out = var1_mlp_out
        self.var2_mlp_in = var2_mlp_in
        self.var2_rnn = var2_rnn
        self.var2_mlp_out = var2_mlp_out
        self.dec_mlp_attn = dec_mlp_attn
        self.dec_mlp_self = dec_mlp_self
        self.dec_mlp_pred = dec_mlp_pred
        # create a shared variable switch for controlling sampling
        ones_ary = numpy.ones((1,)).astype(theano.config.floatX)
        self.train_switch = theano.shared(value=ones_ary, name='train_switch')
        # create shared variables for controlling KLd terms
        self.lam_kld_q2p = theano.shared(value=ones_ary, name='lam_kld_q2p')
        self.lam_kld_p2q = theano.shared(value=ones_ary, name='lam_kld_p2q')
        self.lam_kld_amu = theano.shared(value=ones_ary, name='lam_kld_amu')
        self.lam_kld_alv = theano.shared(value=ones_ary, name='lam_kld_alv')
        self.set_lam_kld(lam_kld_q2p=1.0, lam_kld_p2q=0.1, \
                         lam_kld_amu=0.0, lam_kld_alv=0.0)
        # create shared variables for controlling optimization/updates
        self.lr = theano.shared(value=0.0001*ones_ary, name='lr')
        self.mom_1 = theano.shared(value=0.9*ones_ary, name='mom_1')
        self.mom_2 = theano.shared(value=0.99*ones_ary, name='mom_2')

        # setup a "null pointer" that will point to the computation graph
        # for this model, which can be built by self.build_model_funcs()...
        self.cg = None

        # record the sub-models around which this model is built
        self.params = []
        self.children = [self.reader_mlp,
                         self.pol1_mlp_in, self.pol1_rnn, self.pol1_mlp_out,
                         self.pol2_mlp_in, self.pol2_rnn, self.pol2_mlp_out,
                         self.var1_mlp_in, self.var1_rnn, self.var1_mlp_out,
                         self.var2_mlp_in, self.var2_rnn, self.var2_mlp_out,
                         self.dec_mlp_attn, self.dec_mlp_self, self.dec_mlp_pred]
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

    def set_lam_kld(self, lam_kld_q2p=0.0, lam_kld_p2q=1.0, \
                    lam_kld_amu=0.0, lam_kld_alv=0.0):
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
        new_lam = zero_ary + lam_kld_amu
        self.lam_kld_amu.set_value(to_fX(new_lam))
        new_lam = zero_ary + lam_kld_alv
        self.lam_kld_alv.set_value(to_fX(new_lam))
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
        # initial z
        z_dim = self.get_dim('z')
        self.z_0 = shared_floatx_nans((1,z_dim), name='z_0')

        # initial state of primary policy LSTM (1)
        hp1_dim = self.get_dim('h_pol1')
        cp1_dim = self.get_dim('c_pol1')
        self.hp1_0 = shared_floatx_nans((1,hp1_dim), name='hp1_0')
        self.cp1_0 = shared_floatx_nans((1,cp1_dim), name='cp1_0')
        add_role(self.hp1_0, PARAMETER)
        add_role(self.cp1_0, PARAMETER)
        # initial state of primary policy LSTM (2)
        hp2_dim = self.get_dim('h_pol2')
        cp2_dim = self.get_dim('c_pol2')
        self.hp2_0 = shared_floatx_nans((1,hp2_dim), name='hp2_0')
        self.cp2_0 = shared_floatx_nans((1,cp2_dim), name='cp2_0')
        add_role(self.hp2_0, PARAMETER)
        add_role(self.cp2_0, PARAMETER)

        # initial state of guide policy LSTM (1)
        hv1_dim = self.get_dim('h_var1')
        cv1_dim = self.get_dim('c_var1')
        self.hv1_0 = shared_floatx_nans((1,hv1_dim), name='hv1_0')
        self.cv1_0 = shared_floatx_nans((1,cv1_dim), name='cv1_0')
        add_role(self.hv1_0, PARAMETER)
        add_role(self.cv1_0, PARAMETER)
        # initial state of guide policy LSTM (2)
        hv2_dim = self.get_dim('h_var2')
        cv2_dim = self.get_dim('c_var2')
        self.hv2_0 = shared_floatx_nans((1,hv2_dim), name='hv2_0')
        self.cv2_0 = shared_floatx_nans((1,cv2_dim), name='cv2_0')
        add_role(self.hv2_0, PARAMETER)
        add_role(self.cv2_0, PARAMETER)

        # add the theano shared variables to our parameter lists
        self.params.extend([ self.z_0, \
                             self.hp1_0, self.hp2_0, self.hv1_0, self.hv2_0,
                             self.cp1_0, self.cp2_0, self.cv1_0, self.cv2_0 ])
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
        elif name == 'y':
            return self.y_dim
        elif name == 'nll_scale':
            return 1
        elif name == 'att_spec':
            return self.att_spec_dim
        elif name in ['uin', 'zin']:
            return self.pol1_mlp_out.get_dim('output')
        elif name in ['u', 'z']:
            return self.pol2_mlp_out.get_dim('output')
        elif name in ['h_pol1', 'hp10']:
            return self.pol1_rnn.get_dim('states')
        elif name in ['h_pol2', 'hp20']:
            return self.pol2_rnn.get_dim('states')
        elif name in ['h_var1', 'hv10']:
            return self.var1_rnn.get_dim('states')
        elif name in ['h_var2', 'hv20']:
            return self.var2_rnn.get_dim('states')
        elif name in ['c_pol1', 'cp10']:
            return self.pol1_rnn.get_dim('cells')
        elif name in ['c_pol2', 'cp20']:
            return self.pol2_rnn.get_dim('cells')
        elif name in ['c_var1', 'cv10']:
            return self.var1_rnn.get_dim('cells')
        elif name in ['c_var2', 'cv20']:
            return self.var2_rnn.get_dim('cells')
        elif name in ['nll', 'kl_q2p', 'kl_p2q']:
            return 0
        else:
            super(SeqCondGenALU, self).get_dim(name)
        return

    #------------------------------------------------------------------------

    @recurrent(sequences=['x', 'y', 'u', 'uin', 'nll_scale'], contexts=[],
               states=['z', 'h_pol1', 'c_pol1', 'h_pol2', 'c_pol2', 'h_var1', 'c_var1', 'h_var2', 'c_var2'],
               outputs=['z', 'h_pol1', 'c_pol1', 'h_pol2', 'c_pol2', 'h_var1', 'c_var1', 'h_var2', 'c_var2', 'y_pred', 'nll', 'kl_q2p', 'kl_p2q', 'att_map', 'read_img'])
    def apply(self, x, y, u, uin, nll_scale, z, h_pol1, c_pol1, h_pol2, c_pol2, h_var1, c_var1, h_var2, c_var2):
        # convert z into some "self information" (i.e. recurrent feedback)
        self_info = self.dec_mlp_self.apply(z)
        # convert z into an attention spec
        att_spec = self.dec_mlp_attn.apply(h_pol2)
        if self.use_att:
            # apply the attention-based reader to the input in x
            read_out = self.reader_mlp.apply(x, x, att_spec)
            true_out = self.reader_mlp.apply(y, y, att_spec)
            att_map = self.reader_mlp.att_map(att_spec)
            read_img = self.reader_mlp.write(read_out, att_spec)
            # construct inputs to pol and var nets using attention
            pol1_inp = tensor.concatenate([read_out, att_spec, h_pol2], axis=1)
            var1_inp = tensor.concatenate([y, read_out, att_spec, h_pol2], axis=1)
        else:
            # dummy outputs for attention visualization
            att_map = (0.0 * x) + 0.5
            read_img = x
            # construct inputs to obs and var nets using full observation
            pol1_inp = tensor.concatenate([x, h_pol2], axis=1)
            var1_inp = tensor.concatenate([y, x, h_pol2], axis=1)

        # update the primary policy's first layer LSTM state
        i_pol1 = self.pol1_mlp_in.apply(pol1_inp)
        h_pol1, c_pol1 = self.pol1_rnn.apply(states=h_pol1, cells=c_pol1,
                                             inputs=i_pol1, iterate=False)
        # estimate primary policy's conditional over zin given h_pol1
        p_zin_mean, p_zin_logvar, p_zin = \
                self.pol1_mlp_out.apply(h_pol1, uin)

        if self.use_var:
            # update the guide policy's first layer LSTM state
            i_var1 = self.var1_mlp_in.apply(var1_inp)
            h_var1, c_var1 = self.var1_rnn.apply(states=h_var1, cells=c_var1,
                                                 inputs=i_var1, iterate=False)
            # estimate guide policy's conditional over zin given h_var1
            q_zin_mean, q_zin_logvar, q_zin = \
                    self.var1_mlp_out.apply(h_var1, uin)
        else:
            # use the channel as "deterministic action"
            q_zin_mean, q_zin_logvar, q_zin = p_zin_mean, p_zin_logvar, p_zin
            p_zin = p_zin_mean
            q_zin = p_zin_mean

        # mix samples from p/q based on value of self.train_switch
        _zin = (self.train_switch[0] * q_zin) + \
               ((1.0 - self.train_switch[0]) * p_zin)
        zin = 0.25 * _zin

        # compute KL(guide || primary) and KL(primary || guide)
        kl_q2p_zin = tensor.sum(gaussian_kld(q_zin_mean, q_zin_logvar, \
                                p_zin_mean, p_zin_logvar), axis=1)
        kl_p2q_zin = tensor.sum(gaussian_kld(p_zin_mean, p_zin_logvar, \
                                q_zin_mean, q_zin_logvar), axis=1)



        # update the primary policy's second layer LSTM state
        pol2_inp = tensor.concatenate([zin, h_pol1], axis=1)
        i_pol2 = self.pol2_mlp_in.apply(pol2_inp)
        h_pol2, c_pol2 = self.pol2_rnn.apply(states=h_pol2, cells=c_pol2,
                                             inputs=i_pol2, iterate=False)
        # estimate primary policy's conditional over z given h_pol2
        p_z_mean, p_z_logvar, p_z = \
                self.pol2_mlp_out.apply(h_pol2, u)

        if self.use_var:
            # update the guide policy's second layer LSTM state
            var2_inp = tensor.concatenate([y, zin, h_pol1], axis=1)
            i_var2 = self.var2_mlp_in.apply(var2_inp)
            h_var2, c_var2 = self.var2_rnn.apply(states=h_var2, cells=c_var2,
                                                 inputs=i_var2, iterate=False)
            # estimate guide policy's conditional over z given h_var2
            q_z_mean, q_z_logvar, q_z = \
                    self.var2_mlp_out.apply(h_var2, u)
        else:
            # use the z channel as "deterministic action"
            q_z_mean, q_z_logvar, q_z = p_z_mean, p_z_logvar, p_z
            p_z = p_z_mean
            q_z = p_z_mean

        # mix samples from p/q based on value of self.train_switch
        _z = (self.train_switch[0] * q_z) + \
             ((1.0 - self.train_switch[0]) * p_z)
        z = 0.25 * _z
        # compute KL(guide || primary) and KL(primary || guide)
        kl_q2p_z = tensor.sum(gaussian_kld(q_z_mean, q_z_logvar, \
                              p_z_mean, p_z_logvar), axis=1)
        kl_p2q_z = tensor.sum(gaussian_kld(p_z_mean, p_z_logvar, \
                              q_z_mean, q_z_logvar), axis=1)

        # transform z into an observation/prediction
        y_pred = tensor.nnet.sigmoid(self.dec_mlp_pred.apply(z))

        # compute the NLL of the reconstruction as of this step.
        nll = -nll_scale * tensor.flatten(log_prob_bernoulli(y, y_pred))

        # compute total KL for this step
        kl_q2p = kl_q2p_zin + kl_q2p_z
        kl_p2q = kl_p2q_zin + kl_p2q_z
        return z, h_pol1, c_pol1, h_pol2, c_pol2, h_var1, c_var1, h_var2, c_var2, y_pred, nll, kl_q2p, kl_p2q, att_map, read_img

    #------------------------------------------------------------------------

    @application(inputs=['x', 'y'],
                 outputs=['xs', 'y_preds', 'nlls', 'kl_q2ps', 'kl_p2qs', 'att_maps', 'read_imgs'])
    def process_inputs(self, x, y):
        # get important size and shape information
        u_dim = self.get_dim('u')
        uin_dim = self.get_dim('uin')

        # reshape inputs as necessary
        if self.x_and_y_are_seqs:
            batch_size = x.shape[1]
        else:
            # if we're using "static" inputs, then we need to expand them out
            # into a proper sequential form
            batch_size = x.shape[0]
            x = x.dimshuffle('x',0,1).repeat(self.total_steps, axis=0)
            y = y.dimshuffle('x',0,1).repeat(self.total_steps, axis=0)

        # get initial states for all model components
        z0 = self.z_0.repeat(batch_size, axis=0)
        hp10 = self.hp1_0.repeat(batch_size, axis=0)
        cp10 = self.cp1_0.repeat(batch_size, axis=0)
        hp20 = self.hp2_0.repeat(batch_size, axis=0)
        cp20 = self.cp2_0.repeat(batch_size, axis=0)
        hv10 = self.hv1_0.repeat(batch_size, axis=0)
        cv10 = self.cv1_0.repeat(batch_size, axis=0)
        hv20 = self.hv2_0.repeat(batch_size, axis=0)
        cv20 = self.cv2_0.repeat(batch_size, axis=0)

        # get zero-mean, unit-std. Gaussian noise for use in scan op
        u = self.theano_rng.normal(
                    size=(self.total_steps, batch_size, u_dim),
                    avg=0., std=1.)
        uin = self.theano_rng.normal(
                    size=(self.total_steps, batch_size, uin_dim),
                    avg=0., std=1.)

        # run the multi-stage guided generative process
        _, _, _, _, _, _, _, _, _, y_preds, nlls, kl_q2ps, kl_p2qs, att_maps, read_imgs = \
                self.apply(x=x, y=y, u=u, uin=uin, nll_scale=self.nll_scales,
                           z=z0,
                           h_pol1=hp10, c_pol1=cp10,
                           h_pol2=hp20, c_pol2=cp20,
                           h_var1=hv10, c_var1=cv10,
                           h_var2=hv20, c_var2=cv20)

        # add name tags to the constructed values
        xs = x
        xs.name = "xs"
        y_preds.name = "y_preds"
        nlls.name = "nlls"
        kl_q2ps.name = "kl_q2ps"
        kl_p2qs.name = "kl_p2qs"
        att_maps.name = "att_maps"
        read_imgs.name = "read_imgs"
        return xs, y_preds, nlls, kl_q2ps, kl_p2qs, att_maps, read_imgs

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
        xs, y_preds, nlls, kl_q2ps, kl_p2qs, att_maps, read_imgs = \
                self.process_inputs(x_sym, y_sym)

        # get the expected NLL part of the VFE bound
        self.nll_term = nlls.sum(axis=0).mean()
        self.nll_term.name = "nll_term"

        # get KL(q || p) and KL(p || q) and KL(p || g)
        self.kld_q2p_term = kl_q2ps.sum(axis=0).mean()
        self.kld_q2p_term.name = "kld_q2p_term"
        self.kld_p2q_term = kl_p2qs.sum(axis=0).mean()
        self.kld_p2q_term.name = "kld_p2q_term"

        # grab handles for all the optimizable parameters in our cost
        dummy_cost = self.nll_term + self.kld_q2p_term
        self.cg = ComputationGraph([dummy_cost])
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
        self.joint_updates, applied_updates = get_adam_updates_X( \
                params=self.joint_params, \
                grads=self.joint_grads, alpha=self.lr, \
                beta1=self.mom_1, beta2=self.mom_2, \
                mom2_init=1e-3, smoothing=1e-4, max_grad_norm=10.0)

        # get the total grad norm and (post ADAM scaling) update norm.
        self.grad_norm = sum([tensor.sum(g**2.0) for g in grad_list])
        self.update_norm = sum([tensor.sum(u**2.0) for u in applied_updates])

        # collect the outputs to return from this function
        train_outputs = [self.joint_cost, self.nll_term, \
                   self.kld_q2p_term, self.kld_p2q_term, \
                   self.reg_term, self.grad_norm, self.update_norm]
        bound_outputs = [self.joint_cost, self.nll_term, \
                   self.kld_q2p_term, self.kld_p2q_term, \
                   self.reg_term]
        # collect the required inputs
        inputs = [x_sym, y_sym]

        # compile the theano functions for computing stuff, like for real
        print("Compiling model training/update function...")
        self.train_joint = theano.function(inputs=inputs, \
                                           outputs=train_outputs, \
                                           updates=self.joint_updates)
        print("Compiling model cost estimator function...")
        self.compute_nll_bound = theano.function(inputs=inputs, \
                                                 outputs=bound_outputs)
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
        xs, y_preds, nlls, kl_q2ps, kl_p2qs, att_maps, read_imgs = \
                self.process_inputs(x_sym, y_sym)

        # get the expected NLL part of the VFE bound
        self.nll_term = nlls.sum(axis=0).mean()
        self.nll_term.name = "nll_term"
        # get KL(q || p) and KL(p || q) and KL(p || g)
        self.kld_q2p_term = kl_q2ps.sum(axis=0).mean()
        self.kld_q2p_term.name = "kld_q2p_term"
        self.kld_p2q_term = kl_p2qs.sum(axis=0).mean()
        self.kld_p2q_term.name = "kld_p2q_term"
        # grab handles for all the optimizable parameters in our cost
        dummy_cost = self.nll_term + self.kld_q2p_term
        self.cg = ComputationGraph([dummy_cost])

        # build the function for computing the attention trajectories
        print("Compiling attention tracker...")
        inputs = [x_sym, y_sym]
        outputs = [y_preds, att_maps, read_imgs, xs]
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
            ypreds = outs[0]
            # aggregate sequential attention maps
            obs_dim = self.x_dim
            seq_len = self.total_steps
            samp_count = outs[1].shape[1]
            map_count = int(outs[1].shape[2] / obs_dim)
            a_maps = numpy.zeros((outs[1].shape[0], outs[1].shape[1], obs_dim))
            for m_num in range(map_count):
                start_idx = m_num * obs_dim
                end_idx = (m_num + 1) * obs_dim
                for s1 in range(samp_count):
                    for s2 in range(seq_len):
                        a_maps[s2,s1,:] = a_maps[s2,s1,:] + \
                                          outs[1][s2,s1,start_idx:end_idx]
            # aggregate sequential attention read-outs
            r_outs = numpy.zeros((outs[2].shape[0], outs[2].shape[1], obs_dim))
            for m_num in range(map_count):
                start_idx = m_num * obs_dim
                end_idx = (m_num + 1) * obs_dim
                for s1 in range(samp_count):
                    for s2 in range(seq_len):
                        r_outs[s2,s1,:] = r_outs[s2,s1,:] + \
                                          outs[2][s2,s1,start_idx:end_idx]
            result = [ypreds, a_maps, r_outs, xs]
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
