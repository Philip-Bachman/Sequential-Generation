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