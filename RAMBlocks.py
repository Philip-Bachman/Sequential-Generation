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

from BlocksAttention import ZoomableAttention2d, ZoomableAttention1d
from DKCode import get_adam_updates
from HelperFuncs import constFX, to_fX
from LogPDFs import log_prob_bernoulli, gaussian_kld

###############################################################
###############################################################
## ATTENTION-BASED READERS AND WRITERS, FOR 1D AND 2D INPUTS ##
###############################################################
###############################################################

class SimpleAttentionCore2d(Initializable):
    """
    This class manages a moveable, foveated 2d attention window.

    This class is meant to be wrapped by a "reader" and "writer".

    Parameters:
        x_dim: dimension of "vectorized" inputs
        con_dim: dimension of the controller providing attention params
        height: #rows after reshaping inputs to 2d
        width: #cols after reshaping inputs to 2d
        N: this will be an N x N reader/writer -- N x N at two scales!
        init_scale: the scale of source image vs. attention grid
    """
    def __init__(self, x_dim, con_dim, height, width, N, init_scale, **kwargs):
        super(SimpleAttentionCore2d, self).__init__(**kwargs)

        self.img_height = height
        self.img_width = width
        self.N = N
        self.x_dim = x_dim      # dimension of vectorized image input
        self.con_dim = con_dim  # dimension of controller input
        self.read_dim = 2*N*N   # dimension of reader output
        self.grid_dim = N*N
        self.init_scale = init_scale
        # add and initialize a parameter for controlling sigma scale
        init_ary = (1.75 / self.N) * numpy.ones((1,))
        self.sigma_scale = shared_floatx_nans((1,), name='sigma_scale')
        self.sigma_scale.set_value(init_ary.astype(theano.config.floatX))
        add_role(self.sigma_scale, PARAMETER)

        # get a localized reader mechanism and a controller decoder
        self.zoomer = ZoomableAttention2d(height, width, N, init_scale)
        # con_decoder converts controller input to (5) attention parameters
        self.con_decoder = MLP(activations=[Identity()], dims=[con_dim, 5], \
                               **kwargs)

        # make list of child models (for Blocks stuff)
        self.children = [ self.con_decoder ]
        self.params = [ self.sigma_scale ]
        return

    def get_dim(self, name):
        # Blocks stuff -- not clear when if/and/or how this gets used
        if name == 'h_con':
            return self.con_dim
        elif name in ['x1', 'x2']:
            return self.x_dim
        elif name == 'windows':
            return self.read_dim
        elif name == 'i12':
            return 2*self.x_dim
        elif name == 'r12':
            return self.read_dim
        elif name in ['center_y', 'center_x', 'delta', 'gamma1', 'gamma2']:
            return 0
        else:
            raise ValueError
        return

    def _deltas_and_sigmas(self, delta):
        """
        Get the scaled deltas and sigmas for our foveated attention thing.
        """
        # outer region is at 2x the scale of inner region
        delta1 = 1.0 * delta
        delta2 = 2.0 * delta
        # compute filter bandwidth as linearly proportional to grid scale
        sigma1 = self.sigma_scale[0] * delta1
        sigma2 = self.sigma_scale[0] * delta2
        return delta1, delta2, sigma1, sigma2

    @application(inputs=['x1', 'x2', 'h_con'], outputs=['r12'])
    def read(self, x1, x2, h_con):
        # decode attention parameters from the controller
        l = self.con_decoder.apply(h_con)
        # get base attention parameters
        center_y, center_x, delta, gamma1, gamma2 = self.zoomer.nn2att(l)
        # get deltas and sigmas for our inner/outer attention regions
        delta1, delta2, sigma1, sigma2 = self._deltas_and_sigmas(delta)
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
        l = self.con_decoder.apply(h_con)
        # get base attention parameters
        center_y, center_x, delta, gamma1, gamma2 = self.zoomer.nn2att(l)
        # get deltas and sigmas for our inner/outer attention regions
        delta1, delta2, sigma1, sigma2 = self._deltas_and_sigmas(delta)
        # assume windows are taken from a read operation by this object
        w1 = windows[:,:self.grid_dim]
        w2 = windows[:,self.grid_dim:]
        # perform local write operation at two different scales
        i1 = self.zoomer.write(w1, center_y, center_x, delta1, sigma1)
        i2 = self.zoomer.write(w2, center_y, center_x, delta2, sigma2)
        i12 = tensor.concatenate([i1, i2], axis=1)
        return i12

    @application(inputs=['h_con'], outputs=['i12'])
    def att_map(self, h_con):
        """
        Render a "heat map" of the attention region associated with this
        controller input. Outputs are size (self.img_height, self.img_width).

        Input:
            h_con: controller vector, to be converted by self.con_decoder.
        Output:
            i12: conjoined heat maps for inner and outer foveated regions
        """
        # decode attention parameters from the controller
        l = self.con_decoder.apply(h_con)
        # get base attention parameters
        center_y, center_x, delta, gamma1, gamma2 = self.zoomer.nn2att(l)
        # get deltas and sigmas for our inner/outer attention regions
        delta1, delta2, sigma1, sigma2 = self._deltas_and_sigmas(delta)
        # make a dummy set of "read" responses -- use ones for all pixels
        ones = tensor.alloc(1.0, h_con.shape[0], self.grid_dim)
        # perform local write operation at two different scales
        i1 = gamma1.dimshuffle(0,'x') * \
                self.zoomer.write(ones, center_y, center_x, delta1, sigma1)
        i2 = gamma1.dimshuffle(0,'x') * \
                self.zoomer.write(ones, center_y, center_x, delta2, sigma2)
        i12 = tensor.concatenate([i1, i2], axis=1)
        return i12

    @application(inputs=['x1','center_y','center_x','delta','gamma1','gamma2'], \
                 outputs=['r12'])
    def direct_read(self, x1, center_y, center_x, \
                    delta, gamma1, gamma2):
        # get deltas and sigmas for our inner/outer attention regions
        delta1, delta2, sigma1, sigma2 = self._deltas_and_sigmas(delta)
        # perform local read from x1 at two different scales
        r1 = gamma1.dimshuffle(0,'x') * \
                self.zoomer.read(x1, center_y, center_x, delta1, sigma1)
        r2 = gamma2.dimshuffle(0,'x') * \
                self.zoomer.read(x1, center_y, center_x, delta2, sigma2)
        r12 = tensor.concatenate([r1, r2], axis=1)
        return r12

    @application(inputs=['windows','center_y','center_x','delta'], \
                 outputs=['i12'])
    def direct_write(self, windows, center_y, center_x, delta):
        # get deltas and sigmas for our inner/outer attention regions
        delta1, delta2, sigma1, sigma2 = self._deltas_and_sigmas(delta)
        # assume windows are taken from a read operation by this object
        w1 = windows[:,:self.grid_dim]
        w2 = windows[:,self.grid_dim:]
        # perform local write operation at two different scales
        i1 = self.zoomer.write(w1, center_y, center_x, delta1, sigma1)
        i2 = self.zoomer.write(w2, center_y, center_x, delta2, sigma2)
        i12 = tensor.concatenate([i1, i2], axis=1)
        return i12

    @application(inputs=['center_y','center_x','delta', 'gamma1', 'gamma2'], \
                 outputs=['i12'])
    def direct_att_map(self, center_y, center_x, delta, gamma1, gamma2):
        # get deltas and sigmas for this base delta
        delta1, delta2, sigma1, sigma2 = self._deltas_and_sigmas(delta)
        # make a dummy set of "read" responses -- use ones for all pixels
        ones = tensor.alloc(1.0, center_y.shape[0], self.grid_dim)
        # perform local write operation at two different scales
        i1 = gamma1.dimshuffle(0,'x') * \
                self.zoomer.write(ones, center_y, center_x, delta1, sigma1)
        i2 = gamma1.dimshuffle(0,'x') * \
                self.zoomer.write(ones, center_y, center_x, delta2, sigma2)
        i12 = tensor.concatenate([i1, i2], axis=1)
        return i12


class SimpleAttentionReader2d(SimpleAttentionCore2d):
    """
    This wraps SimpleAttentionCore2d -- for use as a "reader MLP".

    Parameters:
        x_dim: dimension of "vectorized" inputs
        con_dim: dimension of the controller providing attention params
        height: #rows after reshaping inputs to 2d
        width: #cols after reshaping inputs to 2d
        N: this will be an N x N reader -- N x N at two scales!
        init_scale: the scale of source image vs. attention grid
    """
    def __init__(self, x_dim, con_dim, height, width, N, init_scale, **kwargs):
        super(SimpleAttentionReader2d, self).__init__(
                x_dim=x_dim,
                con_dim=con_dim,
                height=height,
                width=width,
                N=N,
                init_scale=init_scale,
                name="reader2d", **kwargs
        )
        return

    def get_dim(self, name):
        # Blocks stuff -- not clear when and how this gets used
        return super(SimpleAttentionReader2d, self).get_dim(name)

    @application(inputs=['x1', 'x2', 'h_con'], outputs=['r12'])
    def apply(self, x1, x2, h_con):
        # apply in a reader performs SimpleAttentionCore2d.read(...)
        r12 = self.read(x1, x2, h_con)
        return r12

class SimpleAttentionWriter2d(SimpleAttentionCore2d):
    """
    This wraps SimpleAttentionCore2d -- for use as a "writer MLP".

    Parameters:
        x_dim: dimension of "vectorized" inputs
        con_dim: dimension of the controller providing attention params
        height: #rows after reshaping inputs to 2d
        width: #cols after reshaping inputs to 2d
        N: this will be an N x N reader -- N x N at two scales!
        init_scale: the scale of source image vs. attention grid
    """
    def __init__(self, x_dim, con_dim, height, width, N, init_scale, **kwargs):
        super(SimpleAttentionWriter2d, self).__init__(
                x_dim=x_dim,
                con_dim=con_dim,
                height=height,
                width=width,
                N=N,
                init_scale=init_scale,
                name="writer2d", **kwargs
        )
        return

    def get_dim(self, name):
        # Blocks stuff -- not clear when and how this gets used
        return super(SimpleAttentionWriter2d, self).get_dim(name)

    @application(inputs=['windows', 'h_con'], outputs=['i12'])
    def apply(self, windows, h_con):
        # apply in a reader performs SimpleAttentionCore2d.read(...)
        i12 = self.write(windows, h_con)
        return i12

#------------------------------------------------------------------------------

class SimpleAttentionCore1d(Initializable):
    """
    This class manages a moveable, foveated 1d attention window.

    Parameters:
        x_dim: dimension of "vectorized" inputs
        con_dim: dimension of the controller providing attention params
        N: this will read N "blob pixels" -- N at each of two scales!
        init_scale: the scale of input cordinates vs. attention grid
    """
    def __init__(self, x_dim, con_dim, N, init_scale, **kwargs):
        super(SimpleAttentionCore1d, self).__init__(**kwargs)
        self.x_dim = x_dim      # dimension of vectorized image input
        self.con_dim = con_dim  # dimension of controller input
        self.N = N              # base attention grid dimension
        self.read_dim = 2*N     # dimension of reader output
        self.init_scale = init_scale # initial scale of input vs. grid
        # add and initialize a parameter for controlling sigma scale
        init_ary = (1.5 / self.N) * numpy.ones((1,))
        self.sigma_scale = shared_floatx_nans((1,), name='sigma_scale')
        self.sigma_scale.set_value(init_ary.astype(theano.config.floatX))
        add_role(self.sigma_scale, PARAMETER)

        # get a localized reader mechanism and a controller decoder
        self.zoomer = ZoomableAttention1d(input_dim=x_dim, N=N, \
                                          init_scale=init_scale)
        # con_decoder converts controller input to (5) attention parameters
        self.con_decoder = MLP(activations=[Identity()], dims=[con_dim, 4], \
                               **kwargs)

        # make list of child models (for Blocks stuff)
        self.children = [ self.con_decoder ]
        self.params = [ self.sigma_scale ]
        return

    def get_dim(self, name):
        # Blocks stuff -- not clear if/when/and/or how this gets used
        if name == 'h_con':
            return self.con_dim
        elif name in ['x1', 'x2']:
            return self.x_dim
        elif name == 'windows':
            return self.read_dim
        elif name == 'i12':
            return 2*self.x_dim
        elif name == 'r12':
            return self.read_dim
        elif name in ['center_x', 'delta', 'gamma1', 'gamma2']:
            return 0
        else:
            raise ValueError
        return

    def _deltas_and_sigmas(self, delta):
        """
        Get the scaled deltas and sigmas for our foveated attention thing.
        """
        # outer region is at 2x the scale of inner region
        delta1 = 1.0 * delta
        delta2 = 2.0 * delta
        # compute filter bandwidth as linearly proportional to grid scale
        sigma1 = self.sigma_scale[0] * delta1
        sigma2 = self.sigma_scale[0] * delta2
        return delta1, delta2, sigma1, sigma2

    @application(inputs=['x1', 'x2', 'h_con'], outputs=['r12'])
    def read(self, x1, x2, h_con):
        # decode attention parameters from the controller
        l = self.con_decoder.apply(h_con)
        # get base attention parameters
        center_x, delta, gamma1, gamma2 = self.zoomer.nn2att(l)
        # get deltas and sigmas for our inner/outer attention regions
        delta1, delta2, sigma1, sigma2 = self._deltas_and_sigmas(delta)
        # perform local read from x1 at two different scales
        r1 = gamma1.dimshuffle(0,'x') * \
                self.zoomer.read(x1, center_x, delta1, sigma1)
        r2 = gamma2.dimshuffle(0,'x') * \
                self.zoomer.read(x1, center_x, delta2, sigma2)
        r12 = tensor.concatenate([r1, r2], axis=1)
        return r12

    @application(inputs=['windows','h_con'], \
                 outputs=['i12'])
    def write(self, windows, h_con):
        # decode attention parameters from the controller
        l = self.con_decoder.apply(h_con)
        # get base attention parameters
        center_x, delta, gamma1, gamma2 = self.zoomer.nn2att(l)
        # get deltas and sigmas for our inner/outer attention regions
        delta1, delta2, sigma1, sigma2 = self._deltas_and_sigmas(delta)
        # assume windows are taken from a read operation by this object
        w1 = windows[:,:self.N]
        w2 = windows[:,self.N:]
        # perform local write operation at two different scales
        i1 = self.zoomer.write(w1, center_x, delta1, sigma1)
        i2 = self.zoomer.write(w2, center_x, delta2, sigma2)
        i12 = tensor.concatenate([i1, i2], axis=1)
        return i12

    @application(inputs=['h_con'], outputs=['i12'])
    def att_map(self, h_con):
        """
        Render a "heat map" of the attention region associated with this
        controller input. Outputs are size (1, self.x_dim).

        Input:
            h_con: controller vector, to be converted by self.con_decoder.
        Output:
            i12: conjoined heat maps for inner and outer foveated regions
        """
        # decode attention parameters from the controller
        l = self.con_decoder.apply(h_con)
        # get base attention parameters
        center_x, delta, gamma1, gamma2 = self.zoomer.nn2att(l)
        # get deltas and sigmas for our inner/outer attention regions
        delta1, delta2, sigma1, sigma2 = self._deltas_and_sigmas(delta)
        # make a dummy set of "read" responses -- use ones for all pixels
        ones = tensor.alloc(1.0, h_con.shape[0], self.N)
        # perform local write operation at two different scales
        i1 = gamma1.dimshuffle(0,'x') * \
                self.zoomer.write(ones, center_x, delta1, sigma1)
        i2 = gamma1.dimshuffle(0,'x') * \
                self.zoomer.write(ones, center_x, delta2, sigma2)
        i12 = tensor.concatenate([i1, i2], axis=1)
        return i12

    @application(inputs=['x1','center_x','delta','gamma1','gamma2'], \
                 outputs=['r12'])
    def direct_read(self, x1, center_x, delta, gamma1, gamma2):
        # get deltas and sigmas for our inner/outer attention regions
        delta1, delta2, sigma1, sigma2 = self._deltas_and_sigmas(delta)
        # perform local read from x1 at two different scales
        r1 = gamma1.dimshuffle(0,'x') * \
                self.zoomer.read(x1, center_x, delta1, sigma1)
        r2 = gamma2.dimshuffle(0,'x') * \
                self.zoomer.read(x1, center_x, delta2, sigma2)
        r12 = tensor.concatenate([r1, r2], axis=1)
        return r12

    @application(inputs=['windows','center_x','delta'], \
                 outputs=['i12'])
    def direct_write(self, windows, center_x, delta):
        # get deltas and sigmas for our inner/outer attention regions
        delta1, delta2, sigma1, sigma2 = self._deltas_and_sigmas(delta)
        # assume windows are taken from a read operation by this object
        w1 = windows[:,:self.N]
        w2 = windows[:,self.N:]
        # perform local write operation at two different scales
        i1 = self.zoomer.write(w1, center_x, delta1, sigma1)
        i2 = self.zoomer.write(w2, center_x, delta2, sigma2)
        i12 = tensor.concatenate([i1, i2], axis=1)
        return i12

    @application(inputs=['center_x','delta','gamma1','gamma2'], \
                 outputs=['i12'])
    def direct_att_map(self, center_x, delta, gamma1, gamma2):
        # get deltas and sigmas for this base delta
        delta1, delta2, sigma1, sigma2 = self._deltas_and_sigmas(delta)
        # make a dummy set of "read" responses -- use ones for all pixels
        ones = tensor.alloc(1.0, center_x.shape[0], self.N)
        # perform local write operation at two different scales
        i1 = gamma1.dimshuffle(0,'x') * \
                self.zoomer.write(ones, center_x, delta1, sigma1)
        i2 = gamma1.dimshuffle(0,'x') * \
                self.zoomer.write(ones, center_x, delta2, sigma2)
        i12 = tensor.concatenate([i1, i2], axis=1)
        return i12


class SimpleAttentionReader1d(SimpleAttentionCore1d):
    """
    This wraps SimpleAttentionCore1d -- for use as a "reader MLP".

    Parameters:
        x_dim: dimension of "vectorized" inputs
        con_dim: dimension of the controller providing attention params
        N: this will be an N-dimensional reader -- N-d  at two scales!
        init_scale: the scale of source image vs. attention grid
    """
    def __init__(self, x_dim, con_dim, N, init_scale, **kwargs):
        super(SimpleAttentionReader1d, self).__init__(
                x_dim=x_dim,
                con_dim=con_dim,
                N=N,
                init_scale=init_scale,
                name="reader1d", **kwargs
        )
        return

    def get_dim(self, name):
        # Blocks stuff -- not clear when and how this gets used
        return super(SimpleAttentionReader1d, self).get_dim(name)

    @application(inputs=['x1', 'x2', 'h_con'], outputs=['r12'])
    def apply(self, x1, x2, h_con):
        # apply in a reader performs SimpleAttentionCore1d.read(...)
        r12 = self.read(x1, x2, h_con)
        return r12

class SimpleAttentionWriter1d(SimpleAttentionCore1d):
    """
    This wraps SimpleAttentionCore1d -- for use as a "writer MLP".

    Parameters:
        x_dim: dimension of "vectorized" inputs
        con_dim: dimension of the controller providing attention params
        N: this will be an N-dimensional reader -- N-d  at two scales!
        init_scale: the scale of source image vs. attention grid
    """
    def __init__(self, x_dim, con_dim, N, init_scale, **kwargs):
        super(SimpleAttentionReader1d, self).__init__(
                x_dim=x_dim,
                con_dim=con_dim,
                N=N,
                init_scale=init_scale,
                name="reader1d", **kwargs
        )
        return

    def get_dim(self, name):
        # Blocks stuff -- not clear when and how this gets used
        return super(SimpleAttentionReader1d, self).get_dim(name)

    @application(inputs=['windows', 'h_con'], outputs=['i12'])
    def apply(self, windows, h_con):
        # apply in a reader performs SimpleAttentionCore1d.read(...)
        i12 = self.write(windows, h_con)
        return i12


##########################################################################
##########################################################################
## Generative model that sequentially constructs sequential predictions ##
##########################################################################
##########################################################################

class OISeqCondGen(BaseRecurrent, Initializable, Random):
    """
    OISeqCondGen -- a model for predicting inputs, given previous inputs.

    For each input in a sequence, this model sequentially builds a prediction
    for the next input. Each of these predictions conditions directly on the
    previous input, and indirectly on even earlier inputs. Conditioning on the
    current input is either "fully informed" or "attention based". Conditioning
    on even earlier inputs is through state that is carried across predictions
    using, e.g., an LSTM.

    Parameters:
        obs_dim: dimension of inputs to observe and predict
        outer_steps: #predictions to make
        inner_steps: #steps when constructing each prediction
        reader_mlp: used for reading from the current input
        writer_mlp: used for writing to prediction of the next input
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
    def __init__(self, obs_dim,
                    outer_steps, inner_steps,
                    reader_mlp, writer_mlp,
                    con_mlp_in, con_rnn,
                    gen_mlp_in, gen_rnn, gen_mlp_out,
                    var_mlp_in, var_rnn, var_mlp_out,
                    mem_mlp_in, mem_rnn, mem_mlp_out,
                    **kwargs):
        super(OISeqCondGen, self).__init__(**kwargs)
        # get shape and length of generative process
        self.obs_dim = obs_dim
        self.outer_steps = outer_steps
        self.inner_steps = inner_steps
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
        # create a shared variable switch for controlling sampling
        ones_ary = numpy.ones((1,)).astype(theano.config.floatX)
        self.train_switch = theano.shared(value=ones_ary, name='train_switch')
        # setup a "null pointer" that will point to the computation graph
        # for this model, which can be built by self.build_model_funcs()...
        self.cg = None

        # record the sub-models used by this OISeqCondGen model
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
        # initialize all OISeqCondGen-owned parameters to zeros...
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
        elif name in ['nll_flag', 'nll', 'kl_q2p', 'kl_p2q']:
            return 0
        else:
            super(OISeqCondGen, self).get_dim(name)
        return

    #------------------------------------------------------------------------

    @recurrent(sequences=['u', 'x_gen', 'x_var', 'nll_flag'], contexts=[],
               states=['c', 'h_mem', 'c_mem', 'h_con', 'c_con', 'h_gen', 'c_gen', 'h_var', 'c_var', 'nll', 'kl_q2p', 'kl_p2q'],
               outputs=['c', 'h_mem', 'c_mem', 'h_con', 'c_con', 'h_gen', 'c_gen', 'h_var', 'c_var', 'nll', 'kl_q2p', 'kl_p2q'])
    def iterate(self, u, x_gen, x_var, nll_flag, c, h_mem, c_mem, h_con, c_con, h_gen, c_gen, h_var, c_var, nll, kl_q2p, kl_p2q):
        # apply reader operation to current "visible" input (i.e. x_gen)
        read_out = self.reader_mlp.apply(x_gen, x_gen, h_con)

        # update the generator RNN state. the generator RNN receives the current
        # prediction, reader output, and controller state as input. these
        # inputs are "preprocessed" through gen_mlp_in.
        i_gen = self.gen_mlp_in.apply( \
                tensor.concatenate([read_out, h_con], axis=1))
        h_gen, c_gen = self.gen_rnn.apply(states=h_gen, cells=c_gen,
                                          inputs=i_gen, iterate=False)
        # update the variational RNN state. the variational RNN receives the
        # NLL gradient, reader output, and controller state as input. these
        # inputs are "preprocessed" through var_mlp_in.
        nll_grad = x_var - tensor.nnet.sigmoid(c)
        i_var = self.var_mlp_in.apply( \
                tensor.concatenate([nll_grad, read_out, h_con], axis=1))
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
        z = (self.train_switch[0] * q_z) + \
            ((1.0 - self.train_switch[0]) * p_z)

        # update the controller RNN state, using sampled z values
        i_con = self.con_mlp_in.apply(tensor.concatenate([z], axis=1))
        h_con, c_con = self.con_rnn.apply(states=h_con, cells=c_con, \
                                          inputs=i_con, iterate=False)

        # update the next input prediction (stored in c)
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
        batch_size = x.shape[1]
        # get input presentation sequences for gen and var models. the var
        # model always gets to look one "outer step" ahead of the gen model.
        #
        # x.shape[0] should be self.outer_steps + 1.
        x = x.repeat(self.inner_steps, axis=0)
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
        x_in = tensor.matrix('x_in')

        # collect symbolic outputs from the model
        cs, h_cons, c_cons, step_nlls, kl_q2ps, kl_p2qs = \
                self.process_inputs(x_in)

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
        self.train_joint = theano.function(inputs=[x_in], \
                                outputs=outputs, updates=self.joint_updates)
        print("Compiling NLL bound estimator function...")
        self.compute_nll_bound = theano.function(inputs=[x_in], \
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


############################################################
############################################################
## ATTENTION-BASED PERCEPTION UNDER TIME CONSTRAINTS      ##
############################################################
############################################################

class SeqCondGen(BaseRecurrent, Initializable, Random):
    """
    SeqCondGen -- constructs conditional densities under time constraints.

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
        con_mlp_out: CondNet for distribution over z given con_rnn
        gen_mlp_in: preprocesses input to the "generator" LSTM
        gen_rnn: the "generator" LSTM
        gen_mlp_out: CondNet for distribution over z given gen_rnn
        var_mlp_in: preprocesses input to the "variational" LSTM
        var_rnn: the "variational" LSTM
        var_mlp_out: CondNet for distribution over z given gen_rnn
    """
    def __init__(self, x_and_y_are_seqs, total_steps, init_steps,
                    exit_rate, nll_weight,
                    step_type, x_dim, y_dim,
                    reader_mlp, writer_mlp,
                    con_mlp_in, con_rnn, con_mlp_out,
                    gen_mlp_in, gen_rnn, gen_mlp_out,
                    var_mlp_in, var_rnn, var_mlp_out,
                    **kwargs):
        super(SeqCondGen, self).__init__(**kwargs)
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
        self.lam_kld_p2g = theano.shared(value=ones_ary, name='lam_kld_p2g')
        self.set_lam_kld(lam_kld_q2p=0.95, lam_kld_p2q=0.05, lam_kld_p2g=0.0)
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

    def set_lam_kld(self, lam_kld_q2p=0.0, lam_kld_p2q=1.0, lam_kld_p2g=0.0):
        """
        Set the relative weight of various KL-divergence terms.

        kld_q2p: KLd between guide reader and primary reader. KL(q||p)
        kld_p2q: KLd between primary reader and guide reader. KL(p||q)
        kld_p2g: KLd between primary reader and primary controller. KL(p||g)
        """
        zero_ary = numpy.zeros((1,))
        new_lam = zero_ary + lam_kld_q2p
        self.lam_kld_q2p.set_value(to_fX(new_lam))
        new_lam = zero_ary + lam_kld_p2q
        self.lam_kld_p2q.set_value(to_fX(new_lam))
        new_lam = zero_ary + lam_kld_p2g
        self.lam_kld_p2g.set_value(to_fX(new_lam))
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
        elif name in ['c', 'c0', 'y']:
            return self.y_dim
        elif name in ['u', 'z']:
            return self.gen_mlp_out.get_dim('output')
        elif name == 'nll_scale':
            return 1
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
            super(SeqCondGen, self).get_dim(name)
        return

    #------------------------------------------------------------------------

    @recurrent(sequences=['x', 'y', 'u', 'u_hc', 'nll_scale'], contexts=[],
               states=['c', 'h_con', 'c_con', 'h_gen', 'c_gen', 'h_var', 'c_var'],
               outputs=['c', 'h_con', 'c_con', 'h_gen', 'c_gen', 'h_var', 'c_var', 'nll', 'kl_q2p', 'kl_p2q', 'kl_p2g', 'att_map', 'read_img'])
    def iterate(self, x, y, u, u_hc, nll_scale, c, h_con, c_con, h_gen, c_gen, h_var, c_var):
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
        att_map = self.reader_mlp.att_map(h_con)
        read_img = self.reader_mlp.write(read_out, h_con)

        # update the primary RNN state
        i_gen = self.gen_mlp_in.apply( \
                tensor.concatenate([read_out, h_con], axis=1))
        h_gen, c_gen = self.gen_rnn.apply(states=h_gen, cells=c_gen,
                                          inputs=i_gen, iterate=False)
        # update the guide RNN state
        nll_grad = y - c_as_y # condition on NLL gradient information
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
        # estimate controller conditional over z given h_con (from time t-1)
        g_z_mean, g_z_logvar, g_z = \
                self.con_mlp_out.apply(h_con, u)

        # mix samples from p/q based on value of self.train_switch
        z = (self.train_switch[0] * q_z) + \
            ((1.0 - self.train_switch[0]) * p_z)

        # update the controller RNN state, using the sampled z values
        i_con = self.con_mlp_in.apply(tensor.concatenate([z], axis=1))
        h_con, c_con = self.con_rnn.apply(states=h_con, cells=c_con, \
                                          inputs=i_con, iterate=False)
        # add a bit of noise to h_con
        h_con = h_con + u_hc
        
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
        kl_p2g = tensor.sum(gaussian_kld(p_z_mean, p_z_logvar, \
                            g_z_mean, g_z_logvar), axis=1)
        return c, h_con, c_con, h_gen, c_gen, h_var, c_var, nll, kl_q2p, kl_p2q, kl_p2g, att_map, read_img

    #------------------------------------------------------------------------

    @application(inputs=['x', 'y'],
                 outputs=['cs', 'h_cons', 'nlls', 'kl_q2ps', 'kl_p2qs', 'kl_p2gs', 'att_maps', 'read_imgs'])
    def process_inputs(self, x, y):
        # get important size and shape information

        z_dim = self.get_dim('z')
        cc_dim = self.get_dim('c_con')
        cg_dim = self.get_dim('c_gen')
        cv_dim = self.get_dim('c_var')
        hc_dim = self.get_dim('h_con')
        hg_dim = self.get_dim('h_gen')
        hv_dim = self.get_dim('h_var')

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
        u_hc0 = 0.1 * self.theano_rng.normal(size=(batch_size, hc_dim),
                                               avg=0., std=1.)
        hc0 = self.hc_0.repeat(batch_size, axis=0) + u_hc0
        cg0 = self.cg_0.repeat(batch_size, axis=0)
        hg0 = self.hg_0.repeat(batch_size, axis=0)
        cv0 = self.cv_0.repeat(batch_size, axis=0)
        hv0 = self.hv_0.repeat(batch_size, axis=0)

        # get zero-mean, unit-std. Gaussian noise for use in scan op
        u = self.theano_rng.normal(
                    size=(self.total_steps, batch_size, z_dim),
                    avg=0., std=1.)

        u_hc = 0.1 * self.theano_rng.normal(
                        size=(self.total_steps, batch_size, hc_dim),
                        avg=0., std=1.)

        # run the multi-stage guided generative process
        cs, h_cons, _, _, _, _, _, nlls, kl_q2ps, kl_p2qs, kl_p2gs, att_maps, read_imgs = \
                self.iterate(x=x, y=y, u=u, u_hc=u_hc,
                             nll_scale=self.nll_scales,
                             c=c0,
                             h_con=hc0, c_con=cc0,
                             h_gen=hg0, c_gen=cg0,
                             h_var=hv0, c_var=cv0)

        # add name tags to the constructed values
        cs.name = "cs"
        h_cons.name = "h_cons"
        nlls.name = "nlls"
        kl_q2ps.name = "kl_q2ps"
        kl_p2qs.name = "kl_p2qs"
        kl_p2gs.name = "kl_p2gs"
        att_maps.name = "att_maps"
        read_imgs.name = "read_imgs"
        return cs, h_cons, nlls, kl_q2ps, kl_p2qs, kl_p2gs, att_maps, read_imgs

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
        cs, h_cons, nlls, kl_q2ps, kl_p2qs, kl_p2gs, att_maps, read_imgs = \
                self.process_inputs(x_sym, y_sym)

        # get the expected NLL part of the VFE bound
        self.nll_term = nlls.sum(axis=0).mean()
        self.nll_term.name = "nll_term"

        # get KL(q || p) and KL(p || q) and KL(p || g)
        self.kld_q2p_term = kl_q2ps.sum(axis=0).mean()
        self.kld_q2p_term.name = "kld_q2p_term"
        self.kld_p2q_term = kl_p2qs.sum(axis=0).mean()
        self.kld_p2q_term.name = "kld_p2q_term"
        self.kld_p2g_term = kl_p2gs.sum(axis=0).mean()
        self.kld_p2g_term.name = "kld_p2g_term"

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
                          (self.lam_kld_p2g[0] * self.kld_p2g_term) + \
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
                   self.kld_q2p_term, self.kld_p2q_term, self.kld_p2g_term, \
                   self.reg_term]
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
        print("Compiling trajectory sampler...")
        self.sample_trajectories = theano.function(inputs=inputs, \
                                                   outputs=[cs, h_cons])
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
        cs, h_cons, nlls, kl_q2ps, kl_p2qs, kl_p2gs, att_maps, read_imgs = \
                self.process_inputs(x_sym, y_sym)
        # build the function for computing the attention trajectories
        print("Compiling attention tracker...")
        inputs = [x_sym, y_sym]
        outputs = [tensor.nnet.sigmoid(cs), att_maps, read_imgs]
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
            return outs
        self.sample_attention = switchy_sampler


        ##############
        # TEMP STUFF #
        ##############
        nll_term = nlls.sum(axis=0).mean()
        kld_q2p_term = kl_q2ps.sum(axis=0).mean()
        # get the proper VFE bound on NLL
        nll_bound = nll_term + kld_q2p_term
        print("Compiling simple NLL bound estimator function...")
        self.simple_nll_bound = theano.function(inputs=[x_sym, y_sym], \
                                outputs=[nll_bound, nll_term, kld_q2p_term])
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
