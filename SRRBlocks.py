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

from BlocksAttention import ZoomableAttentionWindow
from DKCode import get_adam_updates
from HelperFuncs import constFX, to_fX
from LogPDFs import log_prob_bernoulli, gaussian_kld

class AttentionReader2d(Initializable):
    def __init__(self, x_dim, dec_dim, height, width, N, **kwargs):
        super(AttentionReader2d, self).__init__(name="reader", **kwargs)

        self.img_height = height
        self.img_width = width
        self.N = N
        self.x_dim = x_dim
        self.dec_dim = dec_dim
        self.output_dim = 2*N*N

        self.pre_trafo = Linear(
                name=self.name+'_pretrafo',
                input_dim=dec_dim, output_dim=dec_dim,
                weights_init=self.weights_init, biases_init=self.biases_init,
                use_bias=True)
        self.zoomer = ZoomableAttentionWindow(height, width, N)
        self.readout = MLP(activations=[Identity()], dims=[dec_dim, 5], **kwargs)

        self.children = [self.pre_trafo, self.readout]
        return

    def get_dim(self, name):
        if name == 'input':
            return self.dec_dim
        elif name == 'x_dim':
            return self.x_dim
        elif name == 'output':
            return self.output_dim
        else:
            raise ValueError

    @application(inputs=['x', 'x_hat', 'h_dec'], outputs=['r'])
    def apply(self, x, x_hat, h_dec):
        p = self.pre_trafo.apply(h_dec)
        l = self.readout.apply(p)

        center_y, center_x, delta, sigma, gamma = self.zoomer.nn2att(l)

        w     = gamma * self.zoomer.read(x    , center_y, center_x, delta, sigma)
        w_hat = gamma * self.zoomer.read(x_hat, center_y, center_x, delta, sigma)

        return tensor.concatenate([w, w_hat], axis=1)

##########################################################
# LSTM-based sequential revelation and refinement model  #
##########################################################

class SRR_LSTM(BaseRecurrent, Initializable, Random):
    def __init__(self, rev_masks,
                    reader_mlp, writer_mlp,
                    mix_enc_mlp, mix_dec_mlp, mix_var_mlp,
                    enc_mlp_in, enc_rnn, enc_mlp_out,
                    dec_mlp_in, dec_rnn,
                    var_mlp_in, var_rnn, var_mlp_out,
                    **kwargs):
        super(SRR_LSTM, self).__init__(**kwargs)
        # record the desired step count
        self.rev_masks = rev_masks
        # Deal with revelation scheduling
        rmp = self.rev_masks[0].astype(theano.config.floatX)
        rmq = self.rev_masks[1].astype(theano.config.floatX)
        self.rev_masks_p = theano.shared(value=rmp, name='srrm_rev_masks_p')
        self.rev_masks_q = theano.shared(value=rmq, name='srrm_rev_masks_q')
        self.total_steps = self.rev_masks[0].shape[0]
        # grab handles for mixture stuff
        self.mix_enc_mlp = mix_enc_mlp
        self.mix_dec_mlp = mix_dec_mlp
        self.mix_var_mlp = mix_var_mlp
        # grab handles for shared read/write models
        self.reader_mlp = reader_mlp
        self.writer_mlp = writer_mlp
        # grab handles for sequential read/write models
        self.enc_mlp_in = enc_mlp_in
        self.enc_rnn = enc_rnn
        self.enc_mlp_out = enc_mlp_out
        self.dec_mlp_in = dec_mlp_in
        self.dec_rnn = dec_rnn
        self.var_mlp_in = var_mlp_in
        self.var_rnn = var_rnn
        self.var_mlp_out = var_mlp_out
        # setup a "null pointer" that will point to the computation graph
        # for this model, which can be built by self.build_model_funcs()...
        self.cg = None

        # record the sub-models that underlie this model
        self.children = [self.mix_enc_mlp, self.mix_dec_mlp, self.mix_var_mlp,
                         self.reader_mlp, self.writer_mlp,
                         self.enc_mlp_in, self.enc_rnn, self.enc_mlp_out,
                         self.dec_mlp_in, self.dec_rnn,
                         self.var_mlp_in, self.var_rnn, self.var_mlp_out]
        return

    def _allocate(self):
        # allocate shared arrays to hold parameters owned by this model
        c_dim = self.get_dim('c')
        # self.c_0 provides the initial state of the canvas
        self.c_0 = shared_floatx_nans((c_dim,), name='c_0')
        add_role(self.c_0, PARAMETER)
        # add the theano shared variables to our parameter lists
        self.params.extend([ self.c_0 ])
        return

    def _initialize(self):
        # initialize all parameters to zeros...
        for p in self.params:
            p_nan = p.get_value(borrow=False)
            p_zeros = numpy.zeros(p_nan.shape)
            p.set_value(p_zeros.astype(theano.config.floatX))
        return

    def get_dim(self, name):
        if name == 'c':
            return self.reader_mlp.get_dim('x_dim')
        elif name in ['m_p','m_q','mi_p','mi_q']:
            return self.reader_mlp.get_dim('x_dim')
        elif name == 'z_mix':
            return self.mix_enc_mlp.get_dim('output')
        elif name == 'z_gen':
            return self.enc_mlp_out.get_dim('output')
        elif name == 'h_enc':
            return self.enc_rnn.get_dim('states')
        elif name == 'c_enc':
            return self.enc_rnn.get_dim('cells')
        elif name == 'h_dec':
            return self.dec_rnn.get_dim('states')
        elif name == 'c_dec':
            return self.dec_rnn.get_dim('cells')
        elif name == 'h_var':
            return self.var_rnn.get_dim('states')
        elif name == 'c_var':
            return self.var_rnn.get_dim('cells')
        elif name in ['nll', 'kl_q2p', 'kl_p2q']:
            return 0
        else:
            super(SRR_LSTM, self).get_dim(name)
        return

    #------------------------------------------------------------------------

    @recurrent(sequences=['u', 'm_p', 'm_q'], contexts=['x'],
               states=['mi_p', 'mi_q', 'c', 'h_enc', 'c_enc', 'h_dec', 'c_dec', 'h_var', 'c_var', 'nll', 'kl_q2p', 'kl_p2q'],
               outputs=['mi_p', 'mi_q', 'c', 'h_enc', 'c_enc', 'h_dec', 'c_dec', 'h_var', 'c_var', 'nll', 'kl_q2p', 'kl_p2q'])
    def iterate(self, u, m_p, m_q, mi_p, mi_q, c, h_enc, c_enc, h_dec, c_dec, h_var, c_var, nll, kl_q2p, kl_p2q, x):
        # additive steps use c as a "direct workspace", which means it's
        # already directly comparable to x.
        c_as_x = tensor.nnet.sigmoid(c)

        # transform the current belief state into an observation
        full_grad = x - c_as_x

        # get the masked belief state and gradient for primary policy
        xi_for_p = (mi_p * x) + ((1.0 - mi_p) * c_as_x)
        grad_for_p = mi_p * full_grad

        # update the guide policy's revelation mask
        new_to_q = (1.0 - mi_q) * m_q
        mi_q = mi_q + new_to_q
        # get the masked belief state and gradient for guide policy
        xi_for_q = xi_for_p
        grad_for_q = mi_q * full_grad

        # update the guide RNN state
        r_var = self.reader_mlp.apply(xi_for_q, grad_for_q, h_dec)
        i_var = self.var_mlp_in.apply(tensor.concatenate([r_var, h_dec], axis=1))
        h_var, c_var = self.var_rnn.apply(states=h_var, cells=c_var,
                                          inputs=i_var, iterate=False)
        # update the encoder RNN state
        r_enc = self.reader_mlp.apply(xi_for_p, grad_for_p, h_dec)
        i_enc = self.enc_mlp_in.apply(tensor.concatenate([r_enc, h_dec], axis=1))
        h_enc, c_enc = self.enc_rnn.apply(states=h_enc, cells=c_enc,
                                          inputs=i_enc, iterate=False)

        # estimate guide conditional over z given h_var
        q_zg_mean, q_zg_logvar, q_zg = \
                self.var_mlp_out.apply(h_var, u)
        # estimate primary conditional over z given h_enc
        p_zg_mean, p_zg_logvar, p_zg = \
                self.enc_mlp_out.apply(h_enc, u)
        # compute KL(q || p) and KL(p || q) for this step
        kl_q2p = tensor.sum(gaussian_kld(q_zg_mean, q_zg_logvar, \
                            p_zg_mean, p_zg_logvar), axis=1)
        kl_p2q = tensor.sum(gaussian_kld(p_zg_mean, p_zg_logvar, \
                            q_zg_mean, q_zg_logvar), axis=1)

        # update the decoder RNN state, using guidance from the guide
        i_dec = self.dec_mlp_in.apply(tensor.concatenate([q_zg], axis=1))
        h_dec, c_dec = self.dec_rnn.apply(states=h_dec, cells=c_dec, \
                                          inputs=i_dec, iterate=False)

        # update the "workspace" (stored in c)
        c = c + self.writer_mlp.apply(h_dec)

        # update the primary policy's revelation mask
        new_to_p = (1.0 - mi_p) * m_p
        mi_p = mi_p + new_to_p
        # compute the NLL of the reconstruction as of this step
        c_as_x = tensor.nnet.sigmoid(c)
        nll = -1.0 * tensor.flatten(log_prob_bernoulli(x, c_as_x, mask=new_to_p))
        return mi_p, mi_q, c, h_enc, c_enc, h_dec, c_dec, h_var, c_var, nll, kl_q2p, kl_p2q

    #------------------------------------------------------------------------

    @application(inputs=['x'],
                 outputs=['recons', 'step_nlls', 'kl_q2p', 'kl_p2q'])
    def reconstruct(self, x):
        # get important size and shape information
        batch_size = x.shape[0]
        z_mix_dim = self.get_dim('z_mix')
        z_gen_dim = self.get_dim('z_gen')
        ce_dim = self.get_dim('c_enc')
        cd_dim = self.get_dim('c_dec')
        cv_dim = self.get_dim('c_var')
        he_dim = self.get_dim('h_enc')
        hd_dim = self.get_dim('h_dec')
        hv_dim = self.get_dim('h_var')

        # sample zero-mean, unit std. Gaussian noise for mixture init
        u_mix = self.theano_rng.normal(
                    size=(batch_size, z_mix_dim),
                    avg=0., std=1.)
        # transform ZMUV noise based on q(z_mix | x)
        q_zm_mean, q_zm_logvar, q_zm = \
                self.mix_var_mlp.apply(x, u_mix)   # use full x info
        p_zm_mean, p_zm_logvar, p_zm = \
                self.mix_enc_mlp.apply(0.0*x, u_mix) # use masked x info
        # transform samples from q(z_mix | x) into initial generator state
        mix_init = self.mix_dec_mlp.apply(q_zm)
        cd0 = mix_init[:, :cd_dim]
        hd0 = mix_init[:, cd_dim:(cd_dim+hd_dim)]
        ce0 = mix_init[:, (cd_dim+hd_dim):(cd_dim+hd_dim+ce_dim)]
        he0 = mix_init[:, (cd_dim+hd_dim+ce_dim):(cd_dim+hd_dim+ce_dim+he_dim)]
        cv0 = mix_init[:, (cd_dim+hd_dim+ce_dim+he_dim):(cd_dim+hd_dim+ce_dim+he_dim+cv_dim)]
        hv0 = mix_init[:, (cd_dim+hd_dim+ce_dim+he_dim+cv_dim):]

        # compute KL-divergence information for the mixture init step
        kl_q2p_mix = tensor.sum(gaussian_kld(q_zm_mean, q_zm_logvar, \
                                p_zm_mean, p_zm_logvar), axis=1)
        kl_p2q_mix = tensor.sum(gaussian_kld(p_zm_mean, p_zm_logvar, \
                                p_zm_mean, p_zm_logvar), axis=1)
        kl_q2p_mix = kl_q2p_mix.reshape((1, batch_size))
        kl_p2q_mix = kl_p2q_mix.reshape((1, batch_size))

        # get initial state of the reconstruction/imputation
        c0 = tensor.zeros_like(x) + self.c_0
        # get initial mask states for primary and guide policies
        m0_p = tensor.zeros_like(x)
        m0_q = tensor.zeros_like(x)
        # make batch copies of self.rev_masks_p and self.rev_masks_q
        m_p = self.rev_masks_p.dimshuffle(0,'x',1).repeat(batch_size, axis=1)
        m_q = self.rev_masks_q.dimshuffle(0,'x',1).repeat(batch_size, axis=1)

        # get zero-mean, unit-std. Gaussian noise for use in scan op
        u_gen = self.theano_rng.normal(
                    size=(self.total_steps, batch_size, z_gen_dim),
                    avg=0., std=1.)

        # run the multi-stage guided generative process
        _, _, c, _, _, _, _, _, _, step_nlls, kl_q2p_gen, kl_p2q_gen = \
                self.iterate(u=u_gen, m_p=m_p, m_q=m_q, \
                             mi_p=m0_p, mi_q=m0_q, c=c0, \
                             h_enc=he0, c_enc=ce0, \
                             h_dec=hd0, c_dec=cd0, \
                             h_var=hv0, c_var=cv0, \
                             x=x)

        step_nlls.name = "step_nlls"
        # grab the observations generated by the multi-stage process
        recons =  tensor.nnet.sigmoid(c[-1,:,:])
        recons.name = "recons"
        # group up the klds from mixture init and multi-stage generation
        kl_q2p = tensor.vertical_stack(kl_q2p_mix, kl_q2p_gen)
        kl_p2q = tensor.vertical_stack(kl_p2q_mix, kl_p2q_gen)
        kl_q2p.name = "kl_q2p"
        kl_p2q.name = "kl_p2q"
        return recons, step_nlls, kl_q2p, kl_p2q

    def build_model_funcs(self):
        """
        Build the symbolic costs and theano functions relevant to this model.
        """
        # some symbolic vars to represent various inputs/outputs
        self.x_sym = tensor.matrix('x_sym')

        # collect costs for the model
        _, step_nlls, kl_q2p, kl_p2q = self.reconstruct(self.x_sym)

        # get the expected NLL part of the VFE bound
        self.nll_term = step_nlls.sum(axis=0).mean()
        self.nll_term.name = "nll_term"

        # get KL(q || p) and KL(p || q)
        self.kld_q2p_term = kl_q2p.sum(axis=0).mean()
        self.kld_q2p_term.name = "kld_q2p_term"
        self.kld_p2q_term = kl_p2q.sum(axis=0).mean()
        self.kld_p2q_term.name = "kld_p2q_term"

        # get the proper VFE bound on NLL
        self.nll_bound = self.nll_term + self.kld_q2p_term
        self.nll_bound.name = "nll_bound"

        # grab handles for all the optimizable parameters in our cost
        self.cg = ComputationGraph([self.nll_bound])
        self.joint_params = VariableFilter(roles=[PARAMETER])(self.cg.variables)

        # apply some l2 regularization to the model parameters
        self.reg_term = (1e-5 * sum([tensor.sum(p**2.0) for p in self.joint_params]))
        self.reg_term.name = "reg_term"

        # compute the full cost w.r.t. which we will optimize
        self.joint_cost = self.nll_term + (0.95 * self.kld_q2p_term) + \
                          (0.05 * self.kld_p2q_term) + self.reg_term
        self.joint_cost.name = "joint_cost"

        # Get the gradient of the joint cost for all optimizable parameters
        print("Computing gradients of joint_cost...")
        self.joint_grads = OrderedDict()
        grad_list = tensor.grad(self.joint_cost, self.joint_params)
        for i, p in enumerate(self.joint_params):
            self.joint_grads[p] = grad_list[i]

        # shared var learning rate for generator and inferencer
        zero_ary = to_fX( numpy.zeros((1,)) )
        self.lr = theano.shared(value=zero_ary, name='srrb_lr')
        # shared var momentum parameters for generator and inferencer
        self.mom_1 = theano.shared(value=zero_ary, name='srrb_mom_1')
        self.mom_2 = theano.shared(value=zero_ary, name='srrb_mom_2')
        # construct the updates for the generator and inferencer networks
        self.joint_updates = get_adam_updates(params=self.joint_params, \
                grads=self.joint_grads, alpha=self.lr, \
                beta1=self.mom_1, beta2=self.mom_2, \
                mom2_init=1e-4, smoothing=1e-6, max_grad_norm=10.0)

        # collect the outputs to return from this function
        outputs = [self.joint_cost, self.nll_bound, self.nll_term, \
                   self.kld_q2p_term, self.kld_p2q_term, self.reg_term]

        # compile the theano function
        print("Compiling model training/update function...")
        self.train_joint = theano.function(inputs=[self.x_sym], \
                                outputs=outputs, updates=self.joint_updates)
        print("Compiling NLL bound estimator function...")
        self.compute_nll_bound = theano.function(inputs=[self.x_sym], \
                                                 outputs=outputs)
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




##############################################################
##############################################################
## Generative model that constructs images column-by-column ##
##############################################################
##############################################################

class ImgScan(BaseRecurrent, Initializable, Random):
    """
    ImgScan -- a model for predicting image columns, given previous columns.

    For each column in an image, this model sequentially constructs a prediction
    for the next column. Each of these predictions conditions directly on the
    previous column, and indirectly on even earlier columns. Conditioning on the
    current column is either "fully informed" or "attention based". Conditioning
    on even earlier columns is through state that is carried across predictions
    using, e.g., an LSTM.

    Parameters:
        im_shape: [#rows, #cols] shape tuple for input images.
        inner_steps: #steps when constructing each next column's prediction
        reader_mlp: used for reading from the current column
        writer_mlp: used for writing to prediction for the next column
        con_mlp_in: preprocesses input to the "controller" LSTM
        con_rnn: the "controller" LSTM
        gen_mlp_in: preprocesses input to the "generator" LSTM
        gen_rnn: the "generator" LSTM
        gen_mlp_out: CondNet for distribution over z given gen_rnn
        var_mlp_in: preprocesses input to the "variational" LSTM
        var_rnn: the "variational" LSTM
        var_mlp_out: CondNet for distribution over z given gen_rnn
    """
    def __init__(self, im_shape, inner_steps,
                    reader_mlp, writer_mlp,
                    con_mlp_in, con_rnn,
                    gen_mlp_in, gen_rnn, gen_mlp_out,
                    var_mlp_in, var_rnn, var_mlp_out,
                    **kwargs):
        super(ImgScan, self).__init__(**kwargs)
        # get image shape and length of generative process
        self.im_shape = im_shape
        self.obs_dim = im_shape[0]
        self.inner_steps = inner_steps
        self.outer_steps = im_shape[1] - 1 # first column is predicted with a
                                           # constant -- no prediction is made
                                           # for it in the main scan op.
        self.total_steps = self.outer_steps * self.inner_steps
        # construct a sequence of boolean flags for when to measure NLL.
        #   -- we only measure NLL when "next column" becomes "current column"
        self.nll_flags = self._construct_nll_flags()
        # grab handles for shared read/write models
        self.reader_mlp = reader_mlp # reads from current column
        self.writer_mlp = writer_mlp # writes prediction for next column
        # grab handles for controller/generator/variational systems
        self.con_mlp_in = con_mlp_in
        self.con_rnn = con_rnn
        self.gen_mlp_in = gen_mlp_in
        self.gen_rnn = gen_rnn
        self.gen_mlp_out = gen_mlp_out
        self.var_mlp_in = var_mlp_in
        self.var_rnn = var_rnn
        self.var_mlp_out = var_mlp_out
        # setup a "null pointer" that will point to the computation graph
        # for this model, which can be built by self.build_model_funcs()...
        self.cg = None

        # record the sub-models used by this ImgScan model
        self.children = [self.reader_mlp, self.writer_mlp,
                         self.con_mlp_in, self.con_rnn,
                         self.gen_mlp_in, self.gen_rnn, self.gen_mlp_out,
                         self.var_mlp_in, self.var_rnn, self.var_mlp_out]
        return

    def _construct_nll_flags(self):
        """
        Construct shared vector of flags for when to measure NLL.
        """
        nll_flags = shared_floatx_nans((self.total_steps,), name='nll_flags')
        np_flags = numpy.zeros((self.total_steps,))
        for i in range(self.total_steps):
            if ((i + 1) % self.inner_steps) == 0:
                np_flags[i] = 1.0
        nll_flags.set_value(np_flags.astype(theano.config.floatX))
        return nll_flags

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
        self.params.extend([ self.c_0, self.cc_0, self.cg_0, self.cv_0, \
                             self.hc_0, self.hg_0, self.hv_0 ])
        return

    def _initialize(self):
        # initialize all ImgScan-owned parameters to zeros...
        for p in self.params:
            p_nan = p.get_value(borrow=False)
            p_zeros = numpy.zeros(p_nan.shape)
            p.set_value(p_zeros.astype(theano.config.floatX))
        return

    def get_dim(self, name):
        if name == 'c':
            return self.reader_mlp.get_dim('x_dim')
        elif name in ['u', 'z']:
            return self.gen_mlp_out.get_dim('output')
        elif name in ['x_gen', 'x_var']:
            return self.obs_dim
        elif name == 'nll_flag':
            return 1
        elif name == 'h_con':
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
            super(ImgScan, self).get_dim(name)
        return

    #------------------------------------------------------------------------

    @recurrent(sequences=['u', 'x_gen', 'x_var', 'nll_flag'], contexts=[],
               states=['c', 'h_con', 'c_con', 'h_gen', 'c_gen', 'h_var', 'c_var', 'nll', 'kl_q2p', 'kl_p2q'],
               outputs=['c', 'h_con', 'c_con', 'h_gen', 'c_gen', 'h_var', 'c_var', 'nll', 'kl_q2p', 'kl_p2q'])
    def iterate(self, u, x_gen, x_var, nll_flag, c, h_con, c_con, h_gen, c_gen, h_var, c_var, nll, kl_q2p, kl_p2q):
        # transform current prediction into "observation" space
        c_as_x = tensor.nnet.sigmoid(c)
        # compute NLL gradient w.r.t. current prediction
        x_hat = x_var - c_as_x
        # apply reader operation to current "visible" column (i.e. x_gen)
        read_out = self.reader_mlp.apply(x_gen, x_gen, h_con)

        # update the generator RNN state. the generator RNN receives the current
        # prediction, reader output, and controller state as input. these
        # inputs are "preprocessed" through gen_mlp_in.
        i_gen = self.gen_mlp_in.apply( \
                tensor.concatenate([c_as_x, read_out, h_con], axis=1))
        h_gen, c_gen = self.gen_rnn.apply(states=h_gen, cells=c_gen,
                                          inputs=i_gen, iterate=False)
        # update the variational RNN state. the variational RNN receives the
        # NLL gradient, reader output, and controller state as input. these
        # inputs are "preprocessed" through var_mlp_in.
        i_var = self.var_mlp_in.apply( \
                tensor.concatenate([x_hat, read_out, h_con], axis=1))
        h_var, c_var = self.var_rnn.apply(states=h_var, cells=c_var,
                                          inputs=i_var, iterate=False)

        # estimate guide conditional over z given h_var
        q_z_mean, q_z_logvar, q_z = self.var_mlp_out.apply(h_var, u)
        # estimate primary conditional over z given h_enc
        p_z_mean, p_z_logvar, p_z = self.gen_mlp_out.apply(h_gen, u)
        # compute KL(q || p) and KL(p || q) for this step
        kl_q2p = tensor.sum(gaussian_kld(q_z_mean, q_z_logvar, \
                            p_z_mean, p_z_logvar), axis=1)
        kl_p2q = tensor.sum(gaussian_kld(p_z_mean, p_z_logvar, \
                            q_z_mean, q_z_logvar), axis=1)
        # mix samples from p/q based on value of self.train_switch
        #z = (self.train_switch[0] * q_z) + \
        #    ((1.0 - self.train_switch[0]) * p_z)
        z = q_z

        # update the controller RNN state, using sampled z values
        i_con = self.con_mlp_in.apply(tensor.concatenate([z], axis=1))
        h_con, c_con = self.con_rnn.apply(states=h_con, cells=c_con, \
                                          inputs=i_con, iterate=False)

        # update the next column prediction (stored in c)
        c = c + self.writer_mlp.apply(h_con)

        # compute prediction NLL -- but only if nll_flag is "true"
        c_as_x = tensor.nnet.sigmoid(c)
        nll = -nll_flag * tensor.flatten(log_prob_bernoulli(x_var, c_as_x))

        #
        # perform resets and updates that occur only when advancing the "outer"
        # time step (i.e. when the "next column" becomes the "curren column").
        #
        # this includes resetting the prediction in c, updating the "carried"
        # inter-prediction recurrent state, and initializing the con/gen/var
        # intra-prediction recurrent states using the freshly updated
        # inter-prediction recurrent state.
        #
        galf_lln = 1.0 - nll_flag
        # "reset" the prediction workspace c when nll_flag == 1
        c = galf_lln * c
        # compute update for the inter-prediction state, but use gating to only
        # apply the update when nll_flag == 1
        #i_mem = self.mem_mlp_in.apply( \
        #        tensor.concatenate([h_con, c_con], axis=1))
        #h_mem_new, c_mem_new = self.mem_rnn.apply(states=h_men, cells=c_mem, \
        #                                          inputs=i_mem, iterate=False)
        #h_mem = (nll_flag * h_mem_new) + (galf_lln * h_mem)
        #c_mem = (nll_flag * c_mem_new) + (galf_lln * c_mem)
        # compute updates for the controller state info, which gets reset based
        # on h_mem when nll_flag == 1 and stays the same when nll_flag == 0.
        #hc_dim = self.get_dim('h_con')
        #mem_out = self.mem_mlp_out.apply(h_mem)
        #h_con_new = mem_out[:,:hc_dim]
        #c_con_new = mem_out[:,hc_dim:]
        #h_con = (nll_flag * h_con_new) + (galf_lln * h_con)
        #c_con = (nll_flag * c_con_new) + (galf_lln * c_con)
        # reset generator and variational state info when nll_flag == 1
        #h_gen = galf_lln * h_gen
        #c_gen = galf_lln * c_gen
        #h_var = galf_lln * h_var
        #c_var = galf_lln * c_var
        return c, h_con, c_con, h_gen, c_gen, h_var, c_var, nll, kl_q2p, kl_p2q

    #------------------------------------------------------------------------

    @application(inputs=['x'],
                 outputs=['cs', 'h_cons', 'c_cons', 'step_nlls', 'kl_q2ps', 'kl_p2qs'])
    def process_inputs(self, x):
        # get important size and shape information
        batch_size = x.shape[0]
        # reshape inputs and do some dimshuffling
        x = x.reshape((x.shape[0], self.im_shape[0], self.im_shape[1]))
        x = x.repeat(self.inner_steps, axis=2)
        x = x.dimshuffle(2, 0, 1)
        # get column presentation sequences for gen and var models
        x_gen = x[:self.total_steps,:,:]
        x_var = x[self.inner_steps:,:,:]

        # get initial states for all model components
        c0 = self.c_0.repeat(batch_size, axis=0)
        cc0 = self.cc_0.repeat(batch_size, axis=0)
        hc0 = self.hc_0.repeat(batch_size, axis=0)
        cg0 = self.cg_0.repeat(batch_size, axis=0)
        hg0 = self.hg_0.repeat(batch_size, axis=0)
        cv0 = self.cv_0.repeat(batch_size, axis=0)
        hv0 = self.hv_0.repeat(batch_size, axis=0)

        # get zero-mean, unit-std. Gaussian noise for use in scan op
        z_dim = self.get_dim('z')
        u = self.theano_rng.normal(
                    size=(self.total_steps, batch_size, z_dim),
                    avg=0., std=1.)

        # run the multi-stage guided generative process
        cs, h_cons, c_cons, _, _, _, _, step_nlls, kl_q2ps, kl_p2qs = \
                self.iterate(u=u, x_gen=x_gen, x_var=x_var, \
                             nll_flag=self.nll_flags, c=c0, \
                             h_con=hc0, c_con=cc0, \
                             h_gen=hg0, c_gen=cg0, \
                             h_var=hv0, c_var=cv0)

        # attach name tags to the results we're returning
        cs.name = "cs"
        h_cons.name = "h_cons"
        c_cons.name = "c_cons"
        step_nlls.name = "step_nlls"
        kl_q2ps.name = "kl_q2ps"
        kl_p2qs.name = "kl_p2qs"
        return cs, h_cons, c_cons, step_nlls, kl_q2ps, kl_p2qs

    #------------------------------------------------------------------------

    def build_model_funcs(self):
        """
        Build the symbolic costs and theano functions relevant to this model.
        """
        # some symbolic vars to represent various inputs/outputs
        self.x_in = tensor.matrix('x_in')

        # collect symbolic outputs from the model
        cs, h_cons, c_cons, step_nlls, kl_q2ps, kl_p2qs = \
                self.process_inputs(self.x_in)

        # get the expected NLL part of the VFE bound
        self.nll_term = step_nlls.sum(axis=0).mean()
        self.nll_term.name = "nll_term"

        # get KL(q || p) and KL(p || q)
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
        self.joint_cost = self.nll_term + (0.95 * self.kld_q2p_term) + \
                          (0.05 * self.kld_p2q_term) + self.reg_term
        self.joint_cost.name = "joint_cost"

        # get the gradient of the joint cost for all optimizable parameters
        print("Computing gradients of joint_cost...")
        self.joint_grads = OrderedDict()
        grad_list = tensor.grad(self.joint_cost, self.joint_params)
        for i, p in enumerate(self.joint_params):
            self.joint_grads[p] = grad_list[i]

        # shared var learning rate for generator and inferencer
        zero_ary = to_fX( numpy.zeros((1,)) )
        self.lr = theano.shared(value=zero_ary, name='ims_lr')
        # shared var momentum parameters for generator and inferencer
        self.mom_1 = theano.shared(value=zero_ary, name='ims_mom_1')
        self.mom_2 = theano.shared(value=zero_ary, name='ims_mom_2')
        # construct the updates for all trainable parameters
        self.joint_updates = get_adam_updates(params=self.joint_params, \
                grads=self.joint_grads, alpha=self.lr, \
                beta1=self.mom_1, beta2=self.mom_2, \
                mom2_init=1e-4, smoothing=1e-5, max_grad_norm=10.0)

        # collect the outputs to return from this function
        outputs = [self.joint_cost, self.nll_bound, self.nll_term, \
                   self.kld_q2p_term, self.kld_p2q_term, self.reg_term]

        # compile the theano function
        print("Compiling model training/update function...")
        self.train_joint = theano.function(inputs=[self.x_in], \
                                outputs=outputs, updates=self.joint_updates)
        print("Compiling NLL bound estimator function...")
        self.compute_nll_bound = theano.function(inputs=[self.x_in], \
                                                 outputs=outputs)
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
