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

###################################################
###################################################
## ATTENTION-BASED READERS, FOR 1D AND 2D INPUTS ##
###################################################
###################################################

class SimpleAttentionReader2d(Initializable):
    """
    This class manages a moveable, foveated 2d attention window.

    Parameters:
        x_dim: dimension of "vectorized" inputs
        con_dim: dimension of the controller providing attention params
        height: #rows after reshaping inputs to 2d
        width: #cols after reshaping inputs to 2d
        N: this will be an N x N reader -- N x N at two scales!
    """
    def __init__(self, x_dim, con_dim, height, width, N, **kwargs):
        super(SimpleAttentionReader2d, self).__init__(name="reader", **kwargs)

        self.img_height = height
        self.img_width = width
        self.N = N
        self.x_dim = x_dim      # dimension of vectorized image input
        self.con_dim = con_dim  # dimension of controller input
        self.read_dim = 2*N*N   # dimension of reader output
        self.grid_dim = N*N
        # add and initialize a parameter for controlling sigma scale
        init_ary = 0.5 * numpy.ones((1,)).astype(theano.config.floatX)
        self.sigma_scale = shared_floatx_nans((1,), name='sigma_scale')
        self.sigma_scale.set_value(init_ary)
        add_role(self.sigma_scale, PARAMETER)

        # get a localized reader mechanism and a controller decoder
        self.zoomer = ZoomableAttentionWindow(height, width, N)
        self.readout = MLP(activations=[Identity()], dims=[con_dim, 5], **kwargs)

        # make list of child models (for Blocks stuff)
        self.children = [ self.readout ]
        self.params = [ self.sigma_scale ]
        return

    def get_dim(self, name):
        # Blocks stuff -- not clear when and how this gets used
        if name in ['input', 'h_con']:
            return self.con_dim
        elif name in ['x_dim', 'x1', 'x2']:
            return self.x_dim
        elif name in ['output', 'r']:
            return self.read_dim
        else:
            raise ValueError
        return

    @application(inputs=['x1', 'x2', 'h_con'], outputs=['r12'])
    def apply(self, x1, x2, h_con):
        # decode attention parameters from the controller
        l = self.readout.apply(h_con)
        # get base attention parameters
        center_y, center_x, delta, gamma1, gamma2 = self.zoomer.nn2att(l)
        center_y = tensor.zeros_like(center_y) + 14.0
        center_x = tensor.zeros_like(center_x) + 14.0
        delta = tensor.ones_like(delta) * 2.0
        gamma1 = tensor.ones_like(gamma1)
        gamma2 = tensor.ones_like(gamma2)
        # second read-out is at 2x the scale of first read-out
        delta1 = 1.0 * delta
        delta2 = 2.0 * delta
        # compute filter bandwidth as linearly proportional to grid scale
        sigma1 = self.sigma_scale[0] * delta1
        sigma2 = self.sigma_scale[0] * delta2
        # perform local read from x1 at two different scales
        r1 = gamma1.dimshuffle(0,'x') * \
                self.zoomer.read(x1, center_y, center_x, delta1, sigma1)
        r2 = gamma2.dimshuffle(0,'x') * \
                self.zoomer.read(x1, center_y, center_x, delta2, sigma2)
        r12 = tensor.concatenate([r1, r2], axis=1)
        return r12

    @application(inputs=['windows','h_con'], \
                 outputs=['i12'])
    def write(self, windows, h_con):
        # decode attention parameters from the controller
        l = self.readout.apply(h_con)
        # get base attention parameters
        center_y, center_x, delta, gamma1, gamma2 = self.zoomer.nn2att(l)
        center_y = tensor.zeros_like(center_y) + 14.0
        center_x = tensor.zeros_like(center_x) + 14.0
        delta = tensor.ones_like(delta) * 2.0
        gamma1 = tensor.ones_like(gamma1)
        gamma2 = tensor.ones_like(gamma2)
        # second write-out is at 2x the scale of first write-out
        delta1 = 1.0 * delta
        delta2 = 2.0 * delta
        # compute filter bandwidth as linearly proportional to grid scale
        sigma1 = self.sigma_scale[0] * delta1
        sigma2 = self.sigma_scale[0] * delta2
        # assume windows are taken from a read operation by this object
        w1 = windows[:,:self.grid_dim]
        w2 = windows[:,self.grid_dim:]
        # perform local write operation at two different scales
        i1 = self.zoomer.write(w1, center_y, center_x, delta1, sigma1)
        i2 = self.zoomer.write(w2, center_y, center_x, delta2, sigma2)
        i12 = tensor.concatenate([i1, i2], axis=1)
        return i12

    @application(inputs=['h_con'], outputs=['i12', 'i1', 'i2'])
    def att_map(self, h_con):
        """
        Render a "heat map" of the attention region associated with this
        controller input. Outputs are size (self.img_height, self.img_width).

        Input:
            h_con: controller vector, to be converted by self.readout.
        Output:
            i12: heat map for combined inner/outer foveated regions
            i1: heat map for inner region
            i2: heat map for outer region
        """
        # decode attention parameters from the controller
        l = self.readout.apply(h_con)
        # get base attention parameters
        center_y, center_x, delta, gamma1, gamma2 = self.zoomer.nn2att(l)
        center_y = tensor.zeros_like(center_y) + 14.0
        center_x = tensor.zeros_like(center_x) + 14.0
        delta = tensor.ones_like(delta) * 2.0
        gamma1 = tensor.ones_like(gamma1)
        gamma2 = tensor.ones_like(gamma2)
        # make a dummy set of "read" responses -- use ones for all pixels
        ones_window = tensor.alloc(1.0, h_con.shape[0], self.grid_dim)
        # second write-out is at 2x the scale of first write-out
        delta1 = 1.0 * delta
        delta2 = 2.0 * delta
        # compute filter bandwidth as linearly proportional to grid scale
        sigma1 = self.sigma_scale[0] * delta1
        sigma2 = self.sigma_scale[0] * delta2
        # perform local write operation at two different scales
        _i1 = self.zoomer.write(ones_window, center_y, center_x, delta1, sigma1)
        _i2 = self.zoomer.write(ones_window, center_y, center_x, delta2, sigma2)
        i1 = gamma1.dimshuffle(0,'x') * _i1
        i2 = gamma2.dimshuffle(0,'x') * _i2
        i12 = tensor.concatenate([i1, i2], axis=1)
        return i12, i1, i2

    @application(inputs=['im','center_y','center_x','delta','gamma1','gamma2'], \
                 outputs=['r12'])
    def direct_read(self, im, center_y, center_x, \
                    delta, gamma1, gamma2):
        # second read-out is at 2x the scale of first read-out
        delta1 = 1.0 * delta
        delta2 = 2.0 * delta
        # compute filter bandwidth as linearly proportional to grid scale
        sigma1 = self.sigma_scale[0] * delta1
        sigma2 = self.sigma_scale[0] * delta2
        # perform local read from x1 at two different scales
        r1 = gamma1.dimshuffle(0,'x') * self.zoomer.read(im, center_y, center_x, delta1, sigma1)
        r2 = gamma2.dimshuffle(0,'x') * self.zoomer.read(im, center_y, center_x, delta2, sigma2)
        r12 = tensor.concatenate([r1, r2], axis=1)
        return r12

    @application(inputs=['windows','center_y','center_x','delta'], \
                 outputs=['i12'])
    def direct_write(self, windows, center_y, center_x, delta):
        # second write-out is at 2x the scale of first write-out
        delta1 = 1.0 * delta
        delta2 = 2.0 * delta
        # compute filter bandwidth as linearly proportional to grid scale
        sigma1 = self.sigma_scale[0] * delta1
        sigma2 = self.sigma_scale[0] * delta2
        # assume windows are taken from a read operation by this object
        w1 = windows[:,:self.grid_dim]
        w2 = windows[:,self.grid_dim:]
        # perform local write operation at two different scales
        i1 = self.zoomer.write(w1, center_y, center_x, delta1, sigma1)
        i2 = self.zoomer.write(w2, center_y, center_x, delta2, sigma2)
        i12 = tensor.concatenate([i1, i2], axis=1)
        return i12

    @application(inputs=['center_y','center_x','delta', 'gamma1', 'gamma2'], \
                 outputs=['i12', 'i1', 'i2'])
    def direct_att_map(self, center_y, center_x, delta, gamma1, gamma2):
        """
        Render a "heat map" of the attention region associated with this
        controller input. Outputs are size (self.img_height, self.img_width).

        Input:
            center_y: y coordinate of foveated attention grids.
            center_x: x coordinate of foveated attention grids.
            delta: shared scale for foveated attention grids.
            gamma1: (non-negative) amplification for the inner grid
            gamma2: (non-negative) amplification for the outer grid
        Output:
            i12: heat map for combined inner/outer foveated regions
            i1: heat map for inner region
            i2: heat map for outer region
        """
        # make a dummy set of "read" responses -- use ones for all pixels
        ones_window = tensor.alloc(1.0, center_y.shape[0], self.grid_dim)
        # second write-out is at 2x the scale of first write-out
        delta1 = 1.0 * delta
        delta2 = 2.0 * delta
        # compute filter bandwidth as linearly proportional to grid scale
        sigma1 = self.sigma_scale[0] * delta1
        sigma2 = self.sigma_scale[0] * delta2
        # perform local write operation at two different scales
        _i1 = self.zoomer.write(ones_window, center_y, center_x, delta1, sigma1)
        _i2 = self.zoomer.write(ones_window, center_y, center_x, delta2, sigma2)
        i1 = gamma1.dimshuffle(0,'x') * _i1
        i2 = gamma2.dimshuffle(0,'x') * _i2
        i12 = tensor.concatenate([i1, i2], axis=1)
        return i12, i1, i2




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
        mem_mlp_in: preprocesses input to the "memory" LSTM
        mem_rnn: the "memory" LSTM (this stores inter-prediction state)
        mem_mlp_out: emits initial controller state for each prediction
    """
    def __init__(self, im_shape, inner_steps,
                    reader_mlp, writer_mlp,
                    con_mlp_in, con_rnn,
                    gen_mlp_in, gen_rnn, gen_mlp_out,
                    var_mlp_in, var_rnn, var_mlp_out,
                    mem_mlp_in, mem_rnn, mem_mlp_out,
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
        # grab handles for the inter-prediction memory system
        self.mem_mlp_in = mem_mlp_in
        self.mem_rnn = mem_rnn
        self.mem_mlp_out = mem_mlp_out
        # setup a "null pointer" that will point to the computation graph
        # for this model, which can be built by self.build_model_funcs()...
        self.cg = None

        # record the sub-models used by this ImgScan model
        self.children = [self.reader_mlp, self.writer_mlp,
                         self.con_mlp_in, self.con_rnn,
                         self.gen_mlp_in, self.gen_rnn, self.gen_mlp_out,
                         self.var_mlp_in, self.var_rnn, self.var_mlp_out,
                         self.mem_mlp_in, self.mem_rnn, self.mem_mlp_out]
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
        cm_dim = self.get_dim('c_mem')
        hm_dim = self.get_dim('h_mem')
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
        # self.cm_0/self.hm_0 provides initial state of the memory system
        self.cm_0 = shared_floatx_nans((1,cm_dim), name='cm_0')
        add_role(self.cm_0, PARAMETER)
        self.hm_0 = shared_floatx_nans((1,hm_dim), name='hm_0')
        add_role(self.hm_0, PARAMETER)
        # add the theano shared variables to our parameter lists
        self.params.extend([ self.c_0, \
                             self.cc_0, self.cg_0, self.cv_0, self.cm_0, \
                             self.hc_0, self.hg_0, self.hv_0, self.hm_0 ])
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
        elif name == 'h_mem':
            return self.mem_rnn.get_dim('states')
        elif name == 'c_mem':
            return self.mem_rnn.get_dim('cells')
        elif name in ['nll', 'kl_q2p', 'kl_p2q']:
            return 0
        else:
            super(ImgScan, self).get_dim(name)
        return

    #------------------------------------------------------------------------

    @recurrent(sequences=['u', 'x_gen', 'x_var', 'nll_flag'], contexts=[],
               states=['c', 'h_mem', 'c_mem', 'h_con', 'c_con', 'h_gen', 'c_gen', 'h_var', 'c_var', 'nll', 'kl_q2p', 'kl_p2q'],
               outputs=['c', 'h_mem', 'c_mem', 'h_con', 'c_con', 'h_gen', 'c_gen', 'h_var', 'c_var', 'nll', 'kl_q2p', 'kl_p2q'])
    def iterate(self, u, x_gen, x_var, nll_flag, c, h_mem, c_mem, h_con, c_con, h_gen, c_gen, h_var, c_var, nll, kl_q2p, kl_p2q):
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
        i_mem = self.mem_mlp_in.apply( \
                tensor.concatenate([h_con, c_con], axis=1))
        h_mem_new, c_mem_new = self.mem_rnn.apply(states=h_mem, cells=c_mem, \
                                                  inputs=i_mem, iterate=False)
        h_mem = (nll_flag * h_mem_new) + (galf_lln * h_mem)
        c_mem = (nll_flag * c_mem_new) + (galf_lln * c_mem)
        # compute updates for the controller state info, which gets reset based
        # on h_mem when nll_flag == 1 and stays the same when nll_flag == 0.
        hc_dim = self.get_dim('h_con')
        mem_out = self.mem_mlp_out.apply(h_mem)
        h_con_new = mem_out[:,:hc_dim]
        c_con_new = mem_out[:,hc_dim:]
        h_con = (nll_flag * h_con_new) + (galf_lln * h_con)
        c_con = (nll_flag * c_con_new) + (galf_lln * c_con)
        # reset generator and variational state info when nll_flag == 1
        h_gen = galf_lln * h_gen
        c_gen = galf_lln * c_gen
        h_var = galf_lln * h_var
        c_var = galf_lln * c_var
        return c, h_mem, c_mem, h_con, c_con, h_gen, c_gen, h_var, c_var, nll, kl_q2p, kl_p2q

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
        cm0 = self.cm_0.repeat(batch_size, axis=0)
        hm0 = self.hm_0.repeat(batch_size, axis=0)
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
        cs, _, _, h_cons, c_cons, _, _, _, _, step_nlls, kl_q2ps, kl_p2qs = \
                self.iterate(u=u, x_gen=x_gen, x_var=x_var, \
                             nll_flag=self.nll_flags, c=c0, \
                             h_mem=hm0, c_mem=cm0, \
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
                mom2_init=1e-3, smoothing=1e-5, max_grad_norm=10.0)

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

##########################################################
# ATTENTION-BASED PERCEPTION UNDER TIME CONSTRAINTS      #
##########################################################

class SeqCondGen(BaseRecurrent, Initializable, Random):
    """
    SecCondGen -- constructs a conditional density under time constraints.

    This model sequentially constructs a conditional density estimate by taking
    repeated glimpses at the input x, and constructing a hypothesis about the
    output y. The objective is maximum likelihood for (x,y) pairs drawn from
    some training set. We learn a proper generative model, using variational
    inference -- which can be interpreted as a sort of guided policy search.

    Parameters:
        total_steps: total number of steps in sequential estimation process
        init_steps: number of steps we are guaranteed to use
        exit_rate: probability of exiting following each non "init" step
        step_type: whether to use "additive" steps or "jump" steps
        x_dim: dimension of inputs on which to condition
        y_dim: dimension of outputs to predict
        reader_mlp: used for reading from the input
        writer_mlp: used for writing to prediction for the output
        con_mlp_in: preprocesses input to the "controller" LSTM
        con_rnn: the "controller" LSTM
        gen_mlp_in: preprocesses input to the "generator" LSTM
        gen_rnn: the "generator" LSTM
        gen_mlp_out: CondNet for distribution over z given gen_rnn
        var_mlp_in: preprocesses input to the "variational" LSTM
        var_rnn: the "variational" LSTM
        var_mlp_out: CondNet for distribution over z given gen_rnn
    """
    def __init__(self, total_steps, init_steps, exit_rate, \
                    step_type, x_dim, y_dim,
                    reader_mlp, writer_mlp,
                    con_mlp_in, con_rnn,
                    gen_mlp_in, gen_rnn, gen_mlp_out,
                    var_mlp_in, var_rnn, var_mlp_out,
                    **kwargs):
        super(SeqCondGen, self).__init__(**kwargs)
        if not ((step_type == 'add') or (step_type == 'jump')):
            raise ValueError('step_type must be jump or add')
        # record basic structural parameters
        self.total_steps = total_steps
        self.init_steps = init_steps
        self.exit_rate = exit_rate
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
        # grab handles for sequential read/write models
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

        # record the sub-models that underlie this model
        self.children = [self.reader_mlp, self.writer_mlp,
                         self.con_mlp_in, self.con_rnn,
                         self.gen_mlp_in, self.gen_rnn, self.gen_mlp_out,
                         self.var_mlp_in, self.var_rnn, self.var_mlp_out]
        return

    def _construct_nll_scales(self):
        """
        Construct a sequence of scales for weighting NLL measurements.
        """
        nll_scales = shared_floatx_nans((self.total_steps,), name='nll_scales')
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
        elif name in ['c', 'y']:
            return self.y_dim
        elif name in ['u', 'z']:
            return self.gen_mlp_out.get_dim('output')
        elif name == 'nll_scale':
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
            super(SeqCondGen, self).get_dim(name)
        return

    #------------------------------------------------------------------------

    @recurrent(sequences=['u', 'nll_scale'], contexts=['x', 'y'],
               states=['c', 'h_con', 'c_con', 'h_gen', 'c_gen', 'h_var', 'c_var', 'nll', 'kl_q2p', 'kl_p2q'],
               outputs=['c', 'h_con', 'c_con', 'h_gen', 'c_gen', 'h_var', 'c_var', 'nll', 'kl_q2p', 'kl_p2q'])
    def iterate(self, u, nll_scale, c, h_con, c_con, h_gen, c_gen, h_var, c_var, nll, kl_q2p, kl_p2q, x, y):
        if self.step_type == 'add':
            # additive steps use c as a "direct workspace", which means it's
            # already directly comparable to y.
            c_as_y = tensor.nnet.sigmoid(c)
        else:
            # non-additive steps use c_con as a "latent workspace", which means
            # it needs to be transformed before being comparable to y.
            c_as_y = tensor.nnet.sigmoid(self.writer_mlp.apply(c_con))
        # apply the attention-based reader to the input in x
        read_out = self.reader_mlp.apply(x, x, h_con)

        # update the primary RNN state
        i_gen = self.gen_mlp_in.apply( \
                tensor.concatenate([read_out, h_con], axis=1))
        h_gen, c_gen = self.gen_rnn.apply(states=h_gen, cells=c_gen,
                                          inputs=i_gen, iterate=False)
        # update the guide RNN state
        nll_grad = y - c_as_y
        i_var = self.var_mlp_in.apply( \
                tensor.concatenate([nll_grad, read_out, h_con], axis=1))
        h_var, c_var = self.var_rnn.apply(states=h_var, cells=c_var,
                                          inputs=i_var, iterate=False)

        # estimate primary conditional over z given h_gen
        p_z_mean, p_z_logvar, p_z = \
                self.gen_mlp_out.apply(h_gen, u)
        # estimate guide conditional over z given h_var
        q_z_mean, q_z_logvar, q_z = \
                self.var_mlp_out.apply(h_var, u)

        # mix samples from p/q based on value of self.train_switch
        #z = (self.train_switch[0] * q_z) + \
        #    ((1.0 - self.train_switch[0]) * p_z)
        z = q_z

        # update the controller RNN state, using the sampled z values
        i_con = self.con_mlp_in.apply(tensor.concatenate([z], axis=1))
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
        return c, h_con, c_con, h_gen, c_gen, h_var, c_var, nll, kl_q2p, kl_p2q

    #------------------------------------------------------------------------

    @application(inputs=['x', 'y'],
                 outputs=['cs', 'h_cons', 'nlls', 'kl_q2ps', 'kl_p2qs'])
    def process_inputs(self, x, y):
        # get important size and shape information
        batch_size = x.shape[0]
        z_dim = self.get_dim('z')
        cc_dim = self.get_dim('c_con')
        cg_dim = self.get_dim('c_gen')
        cv_dim = self.get_dim('c_var')
        hc_dim = self.get_dim('h_con')
        hg_dim = self.get_dim('h_gen')
        hv_dim = self.get_dim('h_var')

        # get initial states for all model components
        c0 = self.c_0.repeat(batch_size, axis=0)
        cc0 = self.cc_0.repeat(batch_size, axis=0)
        hc0 = self.hc_0.repeat(batch_size, axis=0)
        cg0 = self.cg_0.repeat(batch_size, axis=0)
        hg0 = self.hg_0.repeat(batch_size, axis=0)
        cv0 = self.cv_0.repeat(batch_size, axis=0)
        hv0 = self.hv_0.repeat(batch_size, axis=0)

        # get zero-mean, unit-std. Gaussian noise for use in scan op
        u = self.theano_rng.normal(
                    size=(self.total_steps, batch_size, z_dim),
                    avg=0., std=1.)

        # run the multi-stage guided generative process
        cs, h_cons, _, _, _, _, _, nlls, kl_q2ps, kl_p2qs = \
                self.iterate(u=u, nll_scale=self.nll_scales, c=c0, \
                             h_con=hc0, c_con=cc0, \
                             h_gen=hg0, c_gen=cg0, \
                             h_var=hv0, c_var=cv0, \
                             x=x, y=y)

        # add name tags to the constructed values
        cs.name = "cs"
        h_cons.name = "h_cons"
        nlls.name = "nlls"
        kl_q2ps.name = "kl_q2ps"
        kl_p2qs.name = "kl_p2qs"
        return cs, h_cons, nlls, kl_q2ps, kl_p2qs

    def build_model_funcs(self):
        """
        Build the symbolic costs and theano functions relevant to this model.
        """
        # some symbolic vars to represent various inputs/outputs
        self.x_sym = tensor.matrix('x_sym')
        self.y_sym = tensor.matrix('y_sym')

        # collect estimates of y given x produced by this model
        cs, h_cons, nlls, kl_q2ps, kl_p2qs = \
                self.process_inputs(self.x_sym, self.y_sym)

        # get the expected NLL part of the VFE bound
        self.nll_term = nlls.sum(axis=0).mean()
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
        self.lr = theano.shared(value=zero_ary, name='scg_lr')
        # shared var momentum parameters for generator and inferencer
        self.mom_1 = theano.shared(value=zero_ary, name='scg_mom_1')
        self.mom_2 = theano.shared(value=zero_ary, name='scg_mom_2')
        # construct the updates for all trainable parameters
        self.joint_updates = get_adam_updates(params=self.joint_params, \
                grads=self.joint_grads, alpha=self.lr, \
                beta1=self.mom_1, beta2=self.mom_2, \
                mom2_init=1e-3, smoothing=1e-5, max_grad_norm=10.0)

        # collect the outputs to return from this function
        outputs = [self.joint_cost, self.nll_bound, self.nll_term, \
                   self.kld_q2p_term, self.kld_p2q_term, self.reg_term]
        # collect the required inputs
        inputs = [self.x_sym, self.y_sym]

        # compile the theano functions for computing stuff, like for real
        print("Compiling model training/update function...")
        self.train_joint = theano.function(inputs=inputs, \
                                           outputs=outputs, \
                                           updates=self.joint_updates)
        print("Compiling NLL bound estimator function...")
        self.compute_nll_bound = theano.function(inputs=inputs, \
                                                 outputs=outputs)
        print("Compiling trajectory sampler...")
        self.sample_trajectories = theano.function(inputs=inputs, \
                                                   outputs=[cs, h_cons])
        return

    def build_attention_funcs(self):
        """
        Build functions for playing with this model, under the assumption
        that self.reader_mlp is a SimpleAttentionReader2d.
        """
        # some symbolic vars to represent various inputs/outputs
        x_sym = tensor.matrix('x_sym_att_funcs')
        y_sym = tensor.matrix('y_sym_att_funcs')
        # collect trajectory information from the model
        cs, h_cons, _, _, _ = self.process_inputs(x_sym, y_sym)
        # construct "read-outs" and "heat maps" for the controller trajectories
        # in h_cons and the inputs in self.x_sym.
        xs_list = []
        att_map_list = []
        read_out_list = []
        for i in range(self.total_steps):
            # convert the generated c for this step into x space
            xs_list.append(tensor.nnet.sigmoid(cs[i]))
            # get the attention heat map for this step, for all observations
            # in the input batch
            att_map_i, _, _ = self.reader_mlp.att_map(h_cons[i])
            # get the attention read out for this step, for all observations
            # in the input batch
            raw_read_i = self.reader_mlp.apply(x_sym, x_sym, h_cons[i])
            # get the attention read outs, written back into x space.
            read_out_i = self.reader_mlp.write(raw_read_i, h_cons[i])
            # add the results to the list of per-step results
            att_map_list.append(att_map_i)
            read_out_list.append(read_out_i)
        # stack the per-step result lists into a symbolic theano tensor
        xs_stack = tensor.stack(*xs_list)
        att_map_stack = tensor.stack(*att_map_list)
        read_out_stack = tensor.stack(*read_out_list)
        # build the function for computing the attention trajectories
        print("Compiling attention tracker...")
        inputs = [x_sym, y_sym]
        outputs = [xs_stack, att_map_stack, read_out_stack]
        self.sample_attention = theano.function(inputs=inputs, \
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
