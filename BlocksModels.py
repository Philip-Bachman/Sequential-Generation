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

from BlocksAttention import ZoomableAttention2d
from DKCode import get_adam_updates, get_adam_updates_X
from HelperFuncs import constFX, to_fX, tanh_clip
from LogPDFs import log_prob_bernoulli, gaussian_kld

################################
# Softplus activation function #
################################

class Softplus(Activation):
    @application(inputs=['input_'], outputs=['output'])
    def apply(self, input_):
        return tensor.nnet.softplus(input_)

class BiasedLSTM(BaseRecurrent, Initializable):
    @lazy(allocation=['dim'])
    def __init__(self, dim, ig_bias=0.0, fg_bias=0.0, og_bias=0.0,
                 activation=None, **kwargs):
        super(BiasedLSTM, self).__init__(**kwargs)
        self.dim = dim
        self.ig_bias = constFX(ig_bias) # input gate bias
        self.fg_bias = constFX(fg_bias) # forget gate bias
        self.og_bias = constFX(og_bias) # output gate bias

        if not activation:
            activation = Tanh()
        self.children = [activation]
        return

    def get_dim(self, name):
        if name == 'inputs':
            return self.dim * 4
        if name in ['states', 'cells']:
            return self.dim
        if name == 'mask':
            return 0
        return super(BiasedLSTM, self).get_dim(name)

    def _allocate(self):
        self.W_state = shared_floatx_nans((self.dim, 4*self.dim),
                                          name='W_state')
        self.W_cell_to_in = shared_floatx_nans((self.dim,),
                                               name='W_cell_to_in')
        self.W_cell_to_forget = shared_floatx_nans((self.dim,),
                                                   name='W_cell_to_forget')
        self.W_cell_to_out = shared_floatx_nans((self.dim,),
                                                name='W_cell_to_out')
        add_role(self.W_state, WEIGHT)
        add_role(self.W_cell_to_in, WEIGHT)
        add_role(self.W_cell_to_forget, WEIGHT)
        add_role(self.W_cell_to_out, WEIGHT)

        self.params = [self.W_state, self.W_cell_to_in, self.W_cell_to_forget,
                       self.W_cell_to_out]
        return

    def _initialize(self):
        for w in self.params:
            self.weights_init.initialize(w, self.rng)
        return

    @recurrent(sequences=['inputs', 'mask'], states=['states', 'cells'],
               contexts=[], outputs=['states', 'cells'])
    def apply(self, inputs, states, cells, mask=None):
        """Apply the Long Short Term Memory transition.
        Parameters
        ----------
        states : :class:`~tensor.TensorVariable`
            The 2 dimensional matrix of current states in the shape
            (batch_size, features). Required for `one_step` usage.
        cells : :class:`~tensor.TensorVariable`
            The 2 dimensional matrix of current cells in the shape
            (batch_size, features). Required for `one_step` usage.
        inputs : :class:`~tensor.TensorVariable`
            The 2 dimensional matrix of inputs in the shape (batch_size,
            features * 4).
        mask : :class:`~tensor.TensorVariable`
            A 1D binary array in the shape (batch,) which is 1 if there is
            data available, 0 if not. Assumed to be 1-s only if not given.
        Returns
        -------
        states : :class:`~tensor.TensorVariable`
            Next states of the network.
        cells : :class:`~tensor.TensorVariable`
            Next cell activations of the network.
        """
        def slice_last(x, no):
            return x.T[no*self.dim: (no+1)*self.dim].T
        nonlinearity = self.children[0].apply

        activation = tensor.dot(states, self.W_state) + inputs
        in_gate = tensor.nnet.sigmoid(slice_last(activation, 0) +
                                      (cells * self.W_cell_to_in) +
                                      self.ig_bias)
        forget_gate = tensor.nnet.sigmoid(slice_last(activation, 1) +
                                          (cells * self.W_cell_to_forget) +
                                          self.fg_bias)
        next_cells = (forget_gate * cells +
                      in_gate * nonlinearity(slice_last(activation, 2)))
        out_gate = tensor.nnet.sigmoid(slice_last(activation, 3) +
                                       (next_cells * self.W_cell_to_out) +
                                       self.og_bias)
        next_states = out_gate * nonlinearity(next_cells)

        if mask:
            next_states = (mask[:, None] * next_states +
                           (1 - mask[:, None]) * states)
            next_cells = (mask[:, None] * next_cells +
                          (1 - mask[:, None]) * cells)
        return next_states, next_cells

###################################################
# Diagonal Gaussian conditional density estimator #
###################################################

class CondNet(Initializable, Feedforward):
    """A simple multi-layer perceptron for diagonal Gaussian conditionals.

    Note -- For now, we require both activations and dims to be specified.

    Parameters
    ----------
    activations : list of :class:`.Brick`, :class:`.BoundApplication`,
                  or ``None``
        A list of activations to apply after each linear transformation.
        Give ``None`` to not apply any activation. It is assumed that the
        application method to use is ``apply``. Required for
        :meth:`__init__`. The length of this list should be two less than
        the length of dims, as first dim is the input dim and the last dim
        is the dim of the output Gaussian.
    dims : list of ints
        A list of input dimensions, as well as the output dimension of the
        last layer. Required for :meth:`~.Brick.allocate`.
    """
    def __init__(self, activations=None, dims=None, **kwargs):
        if activations is None:
            raise ValueError("activations must be specified.")
        if dims is None:
            raise ValueError("dims must be specified.")
        if not (len(dims) == (len(activations) + 2)):
            raise ValueError("len(dims) != len(activations) + 2.")
        super(CondNet, self).__init__(**kwargs)

        self.dims = dims
        self.shared_acts = activations

        # construct the shared linear transforms for feedforward
        self.shared_linears = []
        for i in range(len(dims)-2):
            self.shared_linears.append( \
                Linear(dims[i], dims[i+1], name='shared_linear_{}'.format(i)))

        self.mean_linear = Linear(dims[-2], dims[-1], name='mean_linear')
        self.logvar_linear = Linear(dims[-2], dims[-1], name='logvar_linear',
                                    weights_init=Constant(0.))

        self.children = self.shared_linears + self.shared_acts
        self.children.append(self.mean_linear)
        self.children.append(self.logvar_linear)
        return

    def get_dim(self, name):
        if name == 'input':
            return self.dims[0]
        elif name == 'output':
            return self.dims[-1]
        else:
            raise ValueError("Invalid dim name: {}".format(name))
        return

    @property
    def input_dim(self):
        return self.dims[0]

    @property
    def output_dim(self):
        return self.dims[-1]

    @application(inputs=['x', 'u'], outputs=['z_mean', 'z_logvar', 'z'])
    def apply(self, x, u):
        f = [ x ]
        for linear, activation in zip(self.shared_linears, self.shared_acts):
            f.append( activation.apply(linear.apply(f[-1])) )
        z_mean = self.mean_linear.apply(f[-1])
        z_logvar = self.logvar_linear.apply(f[-1])
        z = z_mean + (u * tensor.exp(0.5 * z_logvar))
        return z_mean, z_logvar, z


#-----------------------------------------------------------------------------

class Reader(Initializable):
    def __init__(self, x_dim, dec_dim, **kwargs):
        super(Reader, self).__init__(name="reader", **kwargs)

        self.x_dim = x_dim
        self.dec_dim = dec_dim
        self.output_dim = 2*x_dim

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
        return tensor.concatenate([x, x_hat], axis=1)

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
        self.zoomer = ZoomableAttention2d(height, width, N)
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

#-----------------------------------------------------------------------------

class Writer(Initializable):
    def __init__(self, input_dim, output_dim, **kwargs):
        super(Writer, self).__init__(name="writer", **kwargs)

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.transform = Linear(
                name=self.name+'_transform',
                input_dim=input_dim, output_dim=output_dim,
                weights_init=self.weights_init, biases_init=self.biases_init,
                use_bias=True)

        self.children = [self.transform]

    @application(inputs=['h'], outputs=['c_update'])
    def apply(self, h):
        return self.transform.apply(h)

class AttentionWriter(Initializable):
    def __init__(self, input_dim, output_dim, width, height, N, **kwargs):
        super(AttentionWriter, self).__init__(name="writer", **kwargs)

        self.img_width = width
        self.img_height = height
        self.N = N
        self.input_dim = input_dim
        self.output_dim = output_dim

        assert output_dim == width*height

        self.zoomer = ZoomableAttention2d(height, width, N)

        self.z_trafo = Linear(
                name=self.name+'_ztrafo',
                input_dim=input_dim, output_dim=5,
                weights_init=self.weights_init, biases_init=self.biases_init,
                use_bias=True)

        self.w_trafo = Linear(
                name=self.name+'_wtrafo',
                input_dim=input_dim, output_dim=N*N,
                weights_init=self.weights_init, biases_init=self.biases_init,
                use_bias=True)

        self.children = [self.z_trafo, self.w_trafo]
        return

    @application(inputs=['h'], outputs=['c_update'])
    def apply(self, h):
        w = self.w_trafo.apply(h)
        l = self.z_trafo.apply(h)

        center_y, center_x, delta, sigma, gamma = self.zoomer.nn2att(l)

        c_update = 1./gamma * self.zoomer.write(w, center_y, center_x, delta, sigma)

        return c_update

    @application(inputs=['h'], outputs=['c_update', 'center_y', 'center_x', 'delta'])
    def apply_detailed(self, h):
        w = self.w_trafo.apply(h)
        l = self.z_trafo.apply(h)

        center_y, center_x, delta, sigma, gamma = self.zoomer.nn2att(l)

        c_update = 1./gamma * self.zoomer.write(w, center_y, center_x, delta, sigma)

        return c_update, center_y, center_x, delta

class AttentionWriter2(Initializable):
    def __init__(self, input_dim, output_dim, width, height, N, **kwargs):
        super(AttentionWriter, self).__init__(name="writer", **kwargs)

        self.img_width = width
        self.img_height = height
        self.N = N
        self.input_dim = input_dim
        self.output_dim = output_dim

        assert output_dim == width*height

        self.zoomer = ZoomableAttention2d(height, width, N)

        self.pre_trafo = Linear(
                name=self.name+'_pretrafo',
                input_dim=input_dim, output_dim=input_dim,
                weights_init=self.weights_init, biases_init=self.biases_init,
                use_bias=True)

        self.z_trafo = Linear(
                name=self.name+'_ztrafo',
                input_dim=input_dim, output_dim=5,
                weights_init=self.weights_init, biases_init=self.biases_init,
                use_bias=True)

        self.w_trafo = Linear(
                name=self.name+'_wtrafo',
                input_dim=input_dim, output_dim=N*N,
                weights_init=self.weights_init, biases_init=self.biases_init,
                use_bias=True)

        self.children = [self.pre_trafo, self.z_trafo, self.w_trafo]

    @application(inputs=['h'], outputs=['c_update'])
    def apply(self, h):
        p = self.pre_trafo.apply(h)
        w = self.w_trafo.apply(p)
        l = self.z_trafo.apply(p)

        center_y, center_x, delta, sigma, gamma = self.zoomer.nn2att(l)

        c_update = 1./gamma * self.zoomer.write(w, center_y, center_x, delta, sigma)

        return c_update

    @application(inputs=['h'], outputs=['c_update', 'center_y', 'center_x', 'delta'])
    def apply_detailed(self, h):
        p = self.pre_trafo.apply(h)
        w = self.w_trafo.apply(p)
        l = self.z_trafo.apply(p)

        center_y, center_x, delta, sigma, gamma = self.zoomer.nn2att(l)

        c_update = 1./gamma * self.zoomer.write(w, center_y, center_x, delta, sigma)

        return c_update, center_y, center_x, delta

##########################################################
# Generalized DRAW model, with infinite mixtures and RL. #
#    -- this only works open-loopishly                   #
##########################################################

class IMoOLDrawModels(BaseRecurrent, Initializable, Random):
    def __init__(self, n_iter, step_type, mix_enc_mlp, mix_dec_mlp,
                    reader_mlp, enc_mlp_in, enc_rnn, enc_mlp_out,
                    dec_mlp_in, dec_rnn, dec_mlp_out, writer_mlp,
                    **kwargs):
        super(IMoOLDrawModels, self).__init__(**kwargs)
        if not ((step_type == 'add') or (step_type == 'jump')):
            raise ValueError('step_type must be jump or add')
        # record the desired step count
        self.n_iter = n_iter
        self.step_type = step_type
        # grab handles for mixture stuff
        self.mix_enc_mlp = mix_enc_mlp
        self.mix_dec_mlp = mix_dec_mlp
        # grab handles for IMoOLDRAW model stuff
        self.reader_mlp = reader_mlp
        self.enc_mlp_in = enc_mlp_in
        self.enc_rnn = enc_rnn
        self.enc_mlp_out = enc_mlp_out
        self.dec_mlp_in = dec_mlp_in
        self.dec_rnn = dec_rnn
        self.dec_mlp_out = dec_mlp_out
        self.writer_mlp = writer_mlp
        # regularization noise on RNN states
        zero_ary = to_fX(numpy.zeros((1,)))
        self.rnn_noise = theano.shared(value=zero_ary, name='rnn_noise')

        self.params = []
        # record the sub-models that underlie this model
        self.children = [self.mix_enc_mlp, self.mix_dec_mlp, self.reader_mlp,
                         self.enc_mlp_in, self.enc_rnn, self.enc_mlp_out,
                         self.dec_mlp_in, self.dec_rnn, self.dec_mlp_out,
                         self.writer_mlp]
        return

    def _allocate(self):
        c_dim = self.get_dim('c')
        zm_dim = self.get_dim('z_mix')
        # self.c_0 provides the initial state of the canvas
        self.c_0 = shared_floatx_nans((c_dim,), name='c_0')
        # self.zm_mean provides the mean of z_mix
        self.zm_mean = shared_floatx_nans((zm_dim,), name='zm_mean')
        # self.zm_logvar provides the logvar of z_mix
        self.zm_logvar = shared_floatx_nans((zm_dim,), name='zm_logvar')
        add_role(self.c_0, PARAMETER)
        add_role(self.zm_mean, PARAMETER)
        add_role(self.zm_logvar, PARAMETER)
        # add the theano shared variables to our parameter lists
        self.params.extend([ self.c_0, self.zm_mean, self.zm_logvar ])
        return

    def _initialize(self):
        # initialize to all parameters zeros...
        for p in self.params:
            p_nan = p.get_value(borrow=False)
            p_zeros = numpy.zeros(p_nan.shape)
            p.set_value(p_zeros.astype(theano.config.floatX))
        return

    def get_dim(self, name):
        if name == 'c':
            return self.reader_mlp.get_dim('x_dim')
        elif name == 'z_mix':
            return self.mix_enc_mlp.get_dim('output')
        elif name in ['h_enc','u_enc']:
            return self.enc_rnn.get_dim('states')
        elif name == 'c_enc':
            return self.enc_rnn.get_dim('cells')
        elif name == 'z_gen':
            return self.enc_mlp_out.get_dim('output')
        elif name in ['h_dec','u_dec']:
            return self.dec_rnn.get_dim('states')
        elif name == 'c_dec':
            return self.dec_rnn.get_dim('cells')
        elif name in ['nll', 'kl_q2p', 'kl_p2q']:
            return 0
        elif name == 'center_y':
            return 0
        elif name == 'center_x':
            return 0
        elif name == 'delta':
            return 0
        else:
            super(IMoOLDrawModels, self).get_dim(name)
        return

    def set_rnn_noise(self, rnn_noise=0.0):
        """
        Set the standard deviation of "regularizing noise".
        """
        zero_ary = numpy.zeros((1,))
        new_val = zero_ary + rnn_noise
        self.rnn_noise.set_value(to_fX(new_val))
        return

    #------------------------------------------------------------------------

    @recurrent(sequences=['u', 'u_enc', 'u_dec'], contexts=['x'],
               states=['c', 'h_enc', 'c_enc', 'h_dec', 'c_dec', 'nll', 'kl_q2p', 'kl_p2q'],
               outputs=['c', 'h_enc', 'c_enc', 'h_dec', 'c_dec', 'nll', 'kl_q2p', 'kl_p2q'])
    def apply(self, u, u_enc, u_dec, c, h_enc, c_enc, h_dec, c_dec, nll, kl_q2p, kl_p2q, x):
        # get current prediction
        if self.step_type == 'add':
            # additive steps use c as a "direct workspace", which means it's
            # already directly comparable to x.
            c = c
        else:
            # non-additive steps use c_dec as a "latent workspace", which means
            # it needs to be transformed before being comparable to x.
            c = self.writer_mlp.apply(c_dec)
        c_as_x = tensor.nnet.sigmoid(tanh_clip(c, clip_val=15.0))
        # get the current "reconstruction error"
        x_hat = x - c_as_x
        r_enc = self.reader_mlp.apply(x, x_hat, h_dec)
        # update the encoder RNN state
        i_enc = self.enc_mlp_in.apply(tensor.concatenate([r_enc, h_dec], axis=1))
        h_enc, c_enc = self.enc_rnn.apply(states=h_enc, cells=c_enc,
                                          inputs=i_enc, iterate=False)
        # add noise to the encoder state
        h_enc = h_enc + u_enc
        # estimate encoder conditional over z given h_enc
        q_gen_mean, q_gen_logvar, q_z_gen = \
                self.enc_mlp_out.apply(h_enc, u)
        # estimate decoder conditional over z given h_dec
        p_gen_mean, p_gen_logvar, p_z_gen = \
                self.dec_mlp_out.apply(h_dec, u)
        # update the decoder RNN state
        z_gen = q_z_gen # use samples from q while training
        i_dec = self.dec_mlp_in.apply(tensor.concatenate([z_gen], axis=1))
        h_dec, c_dec = self.dec_rnn.apply(states=h_dec, cells=c_dec, \
                                          inputs=i_dec, iterate=False)
        # add noise to the decoder state
        h_dec = h_dec + u_dec
        # additive steps use c as the "workspace"
        if self.step_type == 'add':
            c = c + self.writer_mlp.apply(h_dec)
        else:
            c = self.writer_mlp.apply(c_dec)
        # compute the NLL of the reconstructiion as of this step
        c_as_x = tensor.nnet.sigmoid(tanh_clip(c, clip_val=15.0))
        nll = -1.0 * tensor.flatten(log_prob_bernoulli(x, c_as_x))
        # compute KL(q || p) and KL(p || q) for this step
        kl_q2p = tensor.sum(gaussian_kld(q_gen_mean, q_gen_logvar, \
                            p_gen_mean, p_gen_logvar), axis=1)
        kl_p2q = tensor.sum(gaussian_kld(p_gen_mean, p_gen_logvar, \
                            q_gen_mean, q_gen_logvar), axis=1)
        return c, h_enc, c_enc, h_dec, c_dec, nll, kl_q2p, kl_p2q

    @recurrent(sequences=['u', 'u_dec'], contexts=[],
               states=['c', 'h_dec', 'c_dec'],
               outputs=['c', 'h_dec', 'c_dec'])
    def decode(self, u, u_dec, c, h_dec, c_dec):
        # sample z from p(z | h_dec) -- we used q(z | h_enc) during training
        p_gen_mean, p_gen_logvar, p_z_gen = \
                self.dec_mlp_out.apply(h_dec, u)
        z_gen = p_z_gen
        # update the decoder RNN state
        i_dec = self.dec_mlp_in.apply(tensor.concatenate([z_gen], axis=1))
        h_dec, c_dec = self.dec_rnn.apply(
                    states=h_dec, cells=c_dec,
                    inputs=i_dec, iterate=False)
        # add noise to decoder state
        h_dec = h_dec + u_dec
        # additive steps use c as the "workspace"
        if self.step_type == 'add':
            c = c + self.writer_mlp.apply(h_dec)
        else:
            c = self.writer_mlp.apply(c_dec)
        return c, h_dec, c_dec

    #------------------------------------------------------------------------

    @application(inputs=['x_in', 'x_out'],
                 outputs=['recons', 'nll', 'kl_q2p', 'kl_p2q'])
    def reconstruct(self, x_in, x_out):
        # get important size and shape information
        batch_size = x_in.shape[0]
        z_mix_dim = self.get_dim('z_mix')
        z_gen_dim = self.get_dim('z_gen')
        ce_dim = self.get_dim('c_enc')
        cd_dim = self.get_dim('c_dec')
        he_dim = self.get_dim('h_enc')
        hd_dim = self.get_dim('h_dec')

        # sample zero-mean, unit std. Gaussian noise for mixture init
        u_mix = self.theano_rng.normal(
                    size=(batch_size, z_mix_dim),
                    avg=0., std=1.)
        # transform ZMUV noise based on q(z_mix | x_in)
        z_mix_mean, z_mix_logvar, z_mix = \
                self.mix_enc_mlp.apply(x_in, u_mix)
        # transform samples from q(z_mix | x_in) into initial generator state
        mix_init = self.mix_dec_mlp.apply(z_mix)
        cd0 = mix_init[:, :cd_dim]
        hd0 = mix_init[:, cd_dim:(cd_dim+hd_dim)]
        ce0 = mix_init[:, (cd_dim+hd_dim):(cd_dim+hd_dim+ce_dim)]
        he0 = mix_init[:, (cd_dim+hd_dim+ce_dim):]
        c0 = tensor.zeros_like(x_out) + self.c_0
        # add noise to initial decoder state
        hd0 = hd0 + (self.rnn_noise[0] * self.theano_rng.normal(
                        size=(hd0.shape[0], hd0.shape[1]),
                        avg=0., std=1.))
        # add noise to initial encoder state
        he0 = he0 + (self.rnn_noise[0] * self.theano_rng.normal(
                        size=(he0.shape[0], hd0.shape[1]),
                        avg=0., std=1.))

        # compute KL-divergence information for the mixture init step
        kl_q2p_mix = tensor.sum(gaussian_kld(z_mix_mean, z_mix_logvar, \
                                self.zm_mean, self.zm_logvar), axis=1)
        kl_p2q_mix = tensor.sum(gaussian_kld(self.zm_mean, self.zm_logvar, \
                                z_mix_mean, z_mix_logvar), axis=1)
        kl_q2p_mix = kl_q2p_mix.reshape((1, batch_size))
        kl_p2q_mix = kl_p2q_mix.reshape((1, batch_size))

        # get zero-mean, unit-std. Gaussian noise for use in scan op
        u_gen = self.theano_rng.normal(
                    size=(self.n_iter, batch_size, z_gen_dim),
                    avg=0., std=1.)
        u_enc = self.rnn_noise[0] * self.theano_rng.normal(
                    size=(self.n_iter, batch_size, he_dim),
                    avg=0., std=1.)
        u_dec = self.rnn_noise[0] * self.theano_rng.normal(
                    size=(self.n_iter, batch_size, hd_dim),
                    avg=0., std=1.)

        # run the multi-stage guided generative process
        c, _, _, _, _, step_nlls, kl_q2p_gen, kl_p2q_gen = \
                self.apply(u=u_gen, u_enc=u_enc, u_dec=u_dec, \
                             c=c0, h_enc=he0, c_enc=ce0, \
                             h_dec=hd0, c_dec=cd0, x=x_out)

        # grab the observations generated by the multi-stage process
        recons = tensor.nnet.sigmoid(tanh_clip(c[-1,:,:], clip_val=15.0))
        recons.name = "recons"
        # get the NLL after the final update for each example
        nll = step_nlls[-1]
        nll.name = "nll"
        # group up the klds from mixture init and multi-stage generation
        kl_q2p = tensor.vertical_stack(kl_q2p_mix, kl_q2p_gen)
        kl_q2p.name = "kl_q2p"
        kl_p2q = tensor.vertical_stack(kl_p2q_mix, kl_p2q_gen)
        kl_p2q.name = "kl_p2q"
        return recons, nll, kl_q2p, kl_p2q

    @application(inputs=['n_samples'], outputs=['x_samples','c_samples'])
    def sample(self, n_samples):
        """Sample from model.

        Returns
        -------

        samples : tensor3 (n_samples, n_iter, x_dim)
        """
        z_mix_dim = self.get_dim('z_mix')
        z_gen_dim = self.get_dim('z_gen')
        cd_dim = self.get_dim('c_dec')
        hd_dim = self.get_dim('h_dec')
        ce_dim = self.get_dim('c_enc')
        he_dim = self.get_dim('h_enc')
        c_dim = self.get_dim('c')

        # sample zero-mean, unit-std. Gaussian noise for the mixture init
        u_mix = self.theano_rng.normal(
                    size=(n_samples, z_mix_dim),
                    avg=0., std=1.)
        # transform noise based on learned mean and logvar
        z_mix = self.zm_mean + (u_mix * tensor.exp(0.5 * self.zm_logvar))
        # transform the sample from p(z_mix) into an initial generator state
        mix_init = self.mix_dec_mlp.apply(z_mix)
        cd0 = mix_init[:, :cd_dim]
        hd0 = mix_init[:, cd_dim:(cd_dim+hd_dim)]
        c0 = tensor.alloc(0.0, n_samples, c_dim) + self.c_0
        # add noise to initial decoder state
        hd0 = hd0 + (self.rnn_noise[0] * self.theano_rng.normal(
                        size=(hd0.shape[0], hd0.shape[1]),
                        avg=0., std=1.))

        # sample from zero-mean unit-std. Gaussian for use in scan op
        u_gen = self.theano_rng.normal(
                    size=(self.n_iter, n_samples, z_gen_dim),
                    avg=0., std=1.)
        u_dec = self.rnn_noise[0] * self.theano_rng.normal(
                    size=(self.n_iter, n_samples, hd_dim),
                    avg=0., std=1.)

        c_samples, _, _, = self.decode(u=u_gen, u_dec=u_dec, \
                                       c=c0, h_dec=hd0, c_dec=cd0)
        x_samples = tensor.nnet.sigmoid(tanh_clip(c_samples, clip_val=15.0))
        return [x_samples, c_samples]

    def build_model_funcs(self):
        """
        Build the symbolic costs and theano functions relevant to this model.
        """
        # some symbolic vars to represent various inputs/outputs
        self.x_in_sym = tensor.matrix('x_in_sym')
        self.x_out_sym = tensor.matrix('x_out_sym')

        # collect reconstructions of x produced by the IMoOLDRAW model
        _, nll, kl_q2p, kl_p2q = self.reconstruct(self.x_in_sym, self.x_out_sym)

        # get the expected NLL part of the VFE bound
        self.nll_term = nll.mean()
        self.nll_term.name = "nll_term"

        # get KL(q || p) and KL(p || q)
        self.kld_q2p_term = kl_q2p.sum(axis=0).mean()
        self.kld_q2p_term.name = "kld_q2p_term"
        self.kld_p2q_term = kl_p2q.sum(axis=0).mean()
        self.kld_p2q_term.name = "kld_p2q_term"
        self.kld_q2p_step = kl_q2p.mean(axis=1)

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
        self.lr = theano.shared(value=zero_ary, name='tbm_lr')
        # shared var momentum parameters for generator and inferencer
        self.mom_1 = theano.shared(value=zero_ary, name='tbm_mom_1')
        self.mom_2 = theano.shared(value=zero_ary, name='tbm_mom_2')
        # construct the updates for the generator and inferencer networks
        self.joint_updates = get_adam_updates(params=self.joint_params, \
                grads=self.joint_grads, alpha=self.lr, \
                beta1=self.mom_1, beta2=self.mom_2, \
                mom2_init=1e-4, smoothing=1e-6, max_grad_norm=10.0)

        # collect the outputs to return from this function
        outputs = [self.joint_cost, self.nll_bound, self.nll_term, \
                   self.kld_q2p_term, self.kld_p2q_term, self.reg_term, \
                   self.kld_q2p_step]

        # compile the theano function
        print("Compiling model training/update function...")
        self.train_joint = theano.function(inputs=[self.x_in_sym, self.x_out_sym], \
                                outputs=outputs, updates=self.joint_updates)
        print("Compiling NLL bound estimator function...")
        self.compute_nll_bound = theano.function(inputs=[self.x_in_sym, self.x_out_sym], \
                                                 outputs=outputs)
        print("Compiling model sampler...")
        n_samples = tensor.iscalar("n_samples")
        x_samples, c_samples = self.sample(n_samples)
        self.do_sample = theano.function([n_samples], \
                                         outputs=[x_samples, c_samples], \
                                         allow_input_downcast=True)
        return

    def build_extra_funcs(self):
        """
        Build functions for computing performance and other stuff.
        """
        # get a list of "bricks" for the variational distributions
        var_bricks = [self.mix_enc_mlp, self.enc_mlp_in, self.enc_rnn,
                      self.enc_mlp_out]

        # grab handles for all the variational parameters in our cost
        cg_vars = self.cg.variables # self.cg should already be built...
        self.var_params = VariableFilter(roles=[PARAMETER], bricks=var_bricks)(cg_vars)

        # get the gradient of the joint cost for all optimizable parameters
        print("Computing gradients of joint_cost (for var params)...")
        self.var_grads = OrderedDict()
        grad_list = tensor.grad(self.joint_cost, self.var_params)
        for i, p in enumerate(self.var_params):
            self.var_grads[p] = grad_list[i]

        # construct a function for training only the variational parameters
        self.var_updates = get_adam_updates(params=self.var_params, \
                grads=self.var_grads, alpha=self.lr, \
                beta1=self.mom_1, beta2=self.mom_2, \
                mom2_init=1e-4, smoothing=1e-6, max_grad_norm=10.0)

        inputs = [self.x_in_sym, self.x_out_sym]
        # collect the outputs to return from this function
        outputs = [self.joint_cost, self.nll_bound, self.nll_term, \
                   self.kld_q2p_term, self.kld_p2q_term, self.reg_term, \
                   self.kld_q2p_step]

        # compile the theano function
        print("Compiling model training/update function (for var params)...")
        self.train_var = theano.function(inputs=inputs, \
                                         outputs=outputs, \
                                         updates=self.var_updates)
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
# Generalized DRAW model, with infinite mixtures and RL. #
#    -- also modified to operate closed-loopishly        #
##########################################################

class IMoCLDrawModels(BaseRecurrent, Initializable, Random):
    def __init__(self, n_iter, step_type,
                    mix_enc_mlp, mix_dec_mlp, mix_var_mlp,
                    reader_mlp, writer_mlp,
                    enc_mlp_in, enc_rnn, enc_mlp_out,
                    dec_mlp_in, dec_rnn,
                    var_mlp_in, var_rnn, var_mlp_out,
                    **kwargs):
        super(IMoCLDrawModels, self).__init__(**kwargs)
        if not ((step_type == 'add') or (step_type == 'jump')):
            raise ValueError('step_type must be jump or add')
        # record the desired step count
        self.n_iter = n_iter
        self.step_type = step_type
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

        self.params = []

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
        elif name == 'center_y':
            return 0
        elif name == 'center_x':
            return 0
        elif name == 'delta':
            return 0
        else:
            super(IMoCLDrawModels, self).get_dim(name)
        return

    #------------------------------------------------------------------------

    @recurrent(sequences=['u'], contexts=['x', 'm'],
               states=['c', 'h_enc', 'c_enc', 'h_dec', 'c_dec', 'h_var', 'c_var', 'nll', 'kl_q2p', 'kl_p2q'],
               outputs=['c', 'h_enc', 'c_enc', 'h_dec', 'c_dec', 'h_var', 'c_var', 'nll', 'kl_q2p', 'kl_p2q'])
    def apply(self, u, c, h_enc, c_enc, h_dec, c_dec, h_var, c_var, nll, kl_q2p, kl_p2q, x, m):
        if self.step_type == 'add':
            # additive steps use c as a "direct workspace", which means it's
            # already directly comparable to x.
            c_as_x = tensor.nnet.sigmoid(c)
        else:
            # non-additive steps use c_dec as a "latent workspace", which means
            # it needs to be transformed before being comparable to x.
            c_as_x = tensor.nnet.sigmoid(self.writer_mlp.apply(c_dec))
        # apply a mask for mixing observed and imputed parts of x. c_as_x
        # gives the current reconstruction of x, for all dimensions. m will
        # use 1 to indicate known values, and 0 to indicate values to impute.
        x_m = (m * x) + ((1.0 - m) * c_as_x) # when m==0 everywhere, this will
                                             # contain no information about x.
        # get the feedback available for use by the guide and primary policy
        x_hat_var = x - c_as_x   # provides LL grad w.r.t. c_as_x everywhere
        x_hat_enc = x_m - c_as_x # provides LL grad w.r.t. c_as_x where m==1
        # update the guide RNN state
        r_var = self.reader_mlp.apply(x, x_hat_var, h_dec)
        i_var = self.var_mlp_in.apply(tensor.concatenate([r_var, h_dec], axis=1))
        h_var, c_var = self.var_rnn.apply(states=h_var, cells=c_var,
                                          inputs=i_var, iterate=False)
        # update the encoder RNN state
        r_enc = self.reader_mlp.apply(x_m, x_hat_enc, h_dec)
        i_enc = self.enc_mlp_in.apply(tensor.concatenate([r_enc, h_dec], axis=1))
        h_enc, c_enc = self.enc_rnn.apply(states=h_enc, cells=c_enc,
                                          inputs=i_enc, iterate=False)
        # estimate guide conditional over z given h_var
        q_zg_mean, q_zg_logvar, q_zg = \
                self.var_mlp_out.apply(h_var, u)
        # estimate primary conditional over z given h_enc
        p_zg_mean, p_zg_logvar, p_zg = \
                self.enc_mlp_out.apply(h_enc, u)
        # update the decoder RNN state, using guidance from the guide
        i_dec = self.dec_mlp_in.apply(tensor.concatenate([q_zg], axis=1))
        #i_dec = self.dec_mlp_in.apply(tensor.concatenate([q_zg, h_enc], axis=1))
        h_dec, c_dec = self.dec_rnn.apply(states=h_dec, cells=c_dec, \
                                          inputs=i_dec, iterate=False)
        # update the "workspace" (stored in c)
        if self.step_type == 'add':
            c = c + self.writer_mlp.apply(h_dec)
        else:
            c = self.writer_mlp.apply(c_dec)
        # compute the NLL of the reconstruction as of this step
        c_as_x = tensor.nnet.sigmoid(c)
        m_inv = 1.0 - m
        nll = -1.0 * tensor.flatten(log_prob_bernoulli(x, c_as_x, mask=m_inv))
        # compute KL(q || p) and KL(p || q) for this step
        kl_q2p = tensor.sum(gaussian_kld(q_zg_mean, q_zg_logvar, \
                            p_zg_mean, p_zg_logvar), axis=1)
        kl_p2q = tensor.sum(gaussian_kld(p_zg_mean, p_zg_logvar, \
                            q_zg_mean, q_zg_logvar), axis=1)
        return c, h_enc, c_enc, h_dec, c_dec, h_var, c_var, nll, kl_q2p, kl_p2q

    @recurrent(sequences=['u'], contexts=['x', 'm'],
               states=['c', 'h_enc', 'c_enc', 'h_dec', 'c_dec'],
               outputs=['c', 'h_enc', 'c_enc', 'h_dec', 'c_dec'])
    def decode(self, u, c, h_enc, c_enc, h_dec, c_dec, x, m):
        # get current state of the reconstruction/imputation
        if self.step_type == 'add':
            c_as_x = tensor.nnet.sigmoid(c)
        else:
            c_as_x = tensor.nnet.sigmoid(self.writer_mlp.apply(c_dec))
        x_m = (m * x) + ((1.0 - m) * c_as_x) # mask the known/imputed vals
        x_hat_enc = x_m - c_as_x             # get feedback used by encoder
        # update the encoder RNN state
        r_enc = self.reader_mlp.apply(x_m, x_hat_enc, h_dec)
        i_enc = self.enc_mlp_in.apply(tensor.concatenate([r_enc, h_dec], axis=1))
        h_enc, c_enc = self.enc_rnn.apply(states=h_enc, cells=c_enc,
                                          inputs=i_enc, iterate=False)
        # estimate primary conditional over z given h_enc
        p_zg_mean, p_zg_logvar, p_zg = \
                self.enc_mlp_out.apply(h_enc, u)
        # update the decoder RNN state, using guidance from the guide
        i_dec = self.dec_mlp_in.apply(tensor.concatenate([p_zg], axis=1))
        #i_dec = self.dec_mlp_in.apply(tensor.concatenate([p_zg, h_enc], axis=1))
        h_dec, c_dec = self.dec_rnn.apply(states=h_dec, cells=c_dec, \
                                          inputs=i_dec, iterate=False)
        # update the "workspace" (stored in c)
        if self.step_type == 'add':
            c = c + self.writer_mlp.apply(h_dec)
        else:
            c = self.writer_mlp.apply(c_dec)
        return c, h_enc, c_enc, h_dec, c_dec

    #------------------------------------------------------------------------

    @application(inputs=['x', 'm'],
                 outputs=['recons', 'nll', 'kl_q2p', 'kl_p2q'])
    def reconstruct(self, x, m):
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

        # get initial state of the reconstruction/imputation
        c0 = tensor.zeros_like(x) + self.c_0
        c_as_x = tensor.nnet.sigmoid(c0)
        x_m = (m * x) + ((1.0 - m) * c_as_x)

        # sample zero-mean, unit std. Gaussian noise for mixture init
        u_mix = self.theano_rng.normal(
                    size=(batch_size, z_mix_dim),
                    avg=0., std=1.)
        # transform ZMUV noise based on q(z_mix | x)
        q_zm_mean, q_zm_logvar, q_zm = \
                self.mix_var_mlp.apply(x, u_mix)   # use full x info
        p_zm_mean, p_zm_logvar, p_zm = \
                self.mix_enc_mlp.apply(x_m, u_mix) # use masked x info
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

        # get zero-mean, unit-std. Gaussian noise for use in scan op
        u_gen = self.theano_rng.normal(
                    size=(self.n_iter, batch_size, z_gen_dim),
                    avg=0., std=1.)

        # run the multi-stage guided generative process
        c, _, _, _, _, _, _, step_nlls, kl_q2p_gen, kl_p2q_gen = \
                self.apply(u=u_gen, c=c0, \
                             h_enc=he0, c_enc=ce0, \
                             h_dec=hd0, c_dec=cd0, \
                             h_var=hv0, c_var=cv0, \
                             x=x, m=m)

        # grab the observations generated by the multi-stage process
        c_as_x = tensor.nnet.sigmoid(c[-1,:,:])
        recons = (m * x) + ((1.0 - m) * c_as_x)
        recons.name = "recons"
        # get the NLL after the final update for each example
        nll = step_nlls[-1]
        nll.name = "nll"
        # group up the klds from mixture init and multi-stage generation
        kl_q2p = tensor.vertical_stack(kl_q2p_mix, kl_q2p_gen)
        kl_q2p.name = "kl_q2p"
        kl_p2q = tensor.vertical_stack(kl_p2q_mix, kl_p2q_gen)
        kl_p2q.name = "kl_p2q"
        return recons, nll, kl_q2p, kl_p2q

    @application(inputs=['x', 'm'], outputs=['recons','c_samples'])
    def sample(self, x, m):
        """
        Sample from model. Sampling can be performed either with or
        without partial control (i.e. conditioning for imputation).

        Returns
        -------

        samples : tensor3 (n_samples, n_iter, x_dim)
        """
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

        # get initial state of the reconstruction/imputation
        c0 = tensor.zeros_like(x) + self.c_0
        c_as_x = tensor.nnet.sigmoid(c0)
        x_m = (m * x) + ((1.0 - m) * c_as_x)

        # sample zero-mean, unit std. Gaussian noise for mixture init
        u_mix = self.theano_rng.normal(
                    size=(batch_size, z_mix_dim),
                    avg=0., std=1.)
        # transform ZMUV noise based on q(z_mix | x)
        p_zm_mean, p_zm_logvar, p_zm = \
                self.mix_enc_mlp.apply(x_m, u_mix) # use masked x info
        # transform samples from q(z_mix | x) into initial generator state
        mix_init = self.mix_dec_mlp.apply(p_zm)
        cd0 = mix_init[:, :cd_dim]
        hd0 = mix_init[:, cd_dim:(cd_dim+hd_dim)]
        ce0 = mix_init[:, (cd_dim+hd_dim):(cd_dim+hd_dim+ce_dim)]
        he0 = mix_init[:, (cd_dim+hd_dim+ce_dim):(cd_dim+hd_dim+ce_dim+he_dim)]
        cv0 = mix_init[:, (cd_dim+hd_dim+ce_dim+he_dim):(cd_dim+hd_dim+ce_dim+he_dim+cv_dim)]
        hv0 = mix_init[:, (cd_dim+hd_dim+ce_dim+he_dim+cv_dim):]

        # get zero-mean, unit-std. Gaussian noise for use in scan op
        u_gen = self.theano_rng.normal(
                    size=(self.n_iter, batch_size, z_gen_dim),
                    avg=0., std=1.)
        # run the sequential generative policy from given initial states
        c_samples, _, _, _, _ = self.decode(u=u_gen, c=c0, h_enc=he0, c_enc=ce0, \
                                            h_dec=hd0, c_dec=cd0, x=x, m=m)
        # convert output into the desired form, and apply masking
        c_as_x = tensor.nnet.sigmoid(c_samples)
        recons = (m * x) + ((1.0 - m) * c_as_x)
        recons.name = "recons"
        return [recons, c_samples]

    def build_model_funcs(self):
        """
        Build the symbolic costs and theano functions relevant to this model.
        """
        # some symbolic vars to represent various inputs/outputs
        self.x_sym = tensor.matrix('x_sym')
        self.m_sym = tensor.matrix('m_sym')

        # collect reconstructions of x produced by the IMoCLDRAW model
        _, nll, kl_q2p, kl_p2q = self.reconstruct(self.x_sym, self.m_sym)

        # get the expected NLL part of the VFE bound
        self.nll_term = nll.mean()
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
        self.joint_cost = self.nll_term + (0.9 * self.kld_q2p_term) + \
                          (0.1 * self.kld_p2q_term) + self.reg_term
        self.joint_cost.name = "joint_cost"

        # Get the gradient of the joint cost for all optimizable parameters
        print("Computing gradients of joint_cost...")
        self.joint_grads = OrderedDict()
        grad_list = tensor.grad(self.joint_cost, self.joint_params)
        for i, p in enumerate(self.joint_params):
            self.joint_grads[p] = grad_list[i]

        # shared var learning rate for generator and inferencer
        zero_ary = to_fX( numpy.zeros((1,)) )
        self.lr = theano.shared(value=zero_ary, name='tbm_lr')
        # shared var momentum parameters for generator and inferencer
        self.mom_1 = theano.shared(value=zero_ary, name='tbm_mom_1')
        self.mom_2 = theano.shared(value=zero_ary, name='tbm_mom_2')
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
        self.train_joint = theano.function(inputs=[self.x_sym, self.m_sym], \
                                outputs=outputs, updates=self.joint_updates)
        print("Compiling NLL bound estimator function...")
        self.compute_nll_bound = theano.function(inputs=[self.x_sym, self.m_sym], \
                                                 outputs=outputs)
        print("Compiling model sampler...")
        x_samples, c_samples = self.sample(self.x_sym, self.m_sym)
        self.do_sample = theano.function([self.x_sym, self.m_sym], \
                                         outputs=[x_samples, c_samples], \
                                         allow_input_downcast=True)
        return

    def build_extra_funcs(self):
        """
        Build functions for computing performance and other stuff.
        """
        # get a list of "bricks" for the variational distributions
        var_bricks = [self.mix_var_mlp, self.var_mlp_in, self.var_rnn,
                      self.var_mlp_out]

        # grab handles for all the variational parameters in our cost
        cg_vars = self.cg.variables # self.cg should already be built...
        self.var_params = VariableFilter(roles=[PARAMETER], bricks=var_bricks)(cg_vars)

        # get the gradient of the joint cost for variational parameters
        print("Computing gradients of joint_cost (for var params)...")
        self.var_grads = OrderedDict()
        grad_list = tensor.grad(self.joint_cost, self.var_params)
        for i, p in enumerate(self.var_params):
            self.var_grads[p] = grad_list[i]

        # construct a function for training only the variational parameters
        self.var_updates = get_adam_updates(params=self.var_params, \
                grads=self.var_grads, alpha=self.lr, \
                beta1=self.mom_1, beta2=self.mom_2, \
                mom2_init=1e-4, smoothing=1e-6, max_grad_norm=10.0)

        inputs = [self.x_sym, self.m_sym]
        # collect the outputs to return from this function
        outputs = [self.joint_cost, self.nll_bound, self.nll_term, \
                   self.kld_q2p_term, self.kld_p2q_term, self.reg_term]

        # compile the theano function
        print("Compiling model training/update function (for var params)...")
        self.train_var = theano.function(inputs=inputs, outputs=outputs, \
                                         updates=self.var_updates)
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


VISUAL_BREAK_STR = """
======================================================================
======================================================================
======================================================================
======================================================================
======================================================================
======================================================================
======================================================================
======================================================================
"""


#######################################################
# Deep DRAW model, adds controller like an RL policy. #
#   -- this only works almost closed-loopishly        #
#######################################################

class RLDrawModel(BaseRecurrent, Initializable, Random):
    def __init__(self, n_iter, step_type, use_pol,
                 reader_mlp, writer_mlp,
                 pol_mlp_in, pol_rnn, pol_mlp_out,
                 enc_mlp_in, enc_rnn, enc_mlp_out,
                 dec_mlp_in, dec_rnn, dec_mlp_out,
                 **kwargs):
        super(RLDrawModel, self).__init__(**kwargs)
        if not ((step_type == 'add') or (step_type == 'jump')):
            raise ValueError('step_type must be jump or add')
        # record the basic model format params
        self.n_iter = n_iter
        self.step_type = step_type
        self.use_pol = use_pol
        # grab handles for submodels
        self.reader_mlp = reader_mlp
        self.writer_mlp = writer_mlp
        self.pol_mlp_in = pol_mlp_in
        self.pol_rnn = pol_rnn
        self.pol_mlp_out = pol_mlp_out
        self.enc_mlp_in = enc_mlp_in
        self.enc_rnn = enc_rnn
        self.enc_mlp_out = enc_mlp_out
        self.dec_mlp_in = dec_mlp_in
        self.dec_rnn = dec_rnn
        self.dec_mlp_out = dec_mlp_out

        # create a shared variable switch for controlling sampling
        ones_ary = numpy.ones((1,)).astype(theano.config.floatX)
        self.train_switch = theano.shared(value=ones_ary, name='train_switch')

        # regularization noise on RNN states
        zero_ary = to_fX(numpy.zeros((1,)))
        self.rnn_noise = theano.shared(value=zero_ary, name='rnn_noise')


        # list for holding references to model params
        self.params = []
        # record the sub-models that underlie this model
        self.children = [self.reader_mlp, self.writer_mlp,
                         self.pol_mlp_in, self.pol_rnn, self.pol_mlp_out,
                         self.enc_mlp_in, self.enc_rnn, self.enc_mlp_out,
                         self.dec_mlp_in, self.dec_rnn, self.dec_mlp_out]
        return

    def _allocate(self):
        """
        Allocate shared parameters used by this model.
        """
        # get size information for the desired parameters
        c_dim = self.get_dim('c')
        cp_dim = self.get_dim('c_pol')
        hp_dim = self.get_dim('h_pol')
        ce_dim = self.get_dim('c_enc')
        he_dim = self.get_dim('h_enc')
        cd_dim = self.get_dim('c_dec')
        hd_dim = self.get_dim('h_dec')
        # self.c_0 provides initial state of the next column prediction
        self.c_0 = shared_floatx_nans((1,c_dim), name='c_0')
        add_role(self.c_0, PARAMETER)
        # self.cp_0/self.hp_0 provides initial state of the primary policy
        self.cp_0 = shared_floatx_nans((1,cp_dim), name='cp_0')
        add_role(self.cp_0, PARAMETER)
        self.hp_0 = shared_floatx_nans((1,hp_dim), name='hp_0')
        add_role(self.hp_0, PARAMETER)
        # self.ce_0/self.he_0 provides initial state of the guide policy
        self.ce_0 = shared_floatx_nans((1,ce_dim), name='ce_0')
        add_role(self.ce_0, PARAMETER)
        self.he_0 = shared_floatx_nans((1,he_dim), name='he_0')
        add_role(self.he_0, PARAMETER)
        # self.cd_0/self.hd_0 provides initial state of the shared dynamics
        self.cd_0 = shared_floatx_nans((1,cd_dim), name='cd_0')
        add_role(self.cd_0, PARAMETER)
        self.hd_0 = shared_floatx_nans((1,hd_dim), name='hd_0')
        add_role(self.hd_0, PARAMETER)
        # add the theano shared variables to our parameter lists
        self.params.extend([ self.c_0,
                             self.cp_0, self.ce_0, self.cd_0,
                             self.hp_0, self.he_0, self.hd_0 ])
        return

    def _initialize(self):
        # initialize to all parameters zeros...
        for p in self.params:
            p_nan = p.get_value(borrow=False)
            p_zeros = numpy.zeros(p_nan.shape)
            p.set_value(p_zeros.astype(theano.config.floatX))
        return

    def get_dim(self, name):
        if name == 'c':
            return self.reader_mlp.get_dim('x_dim')
        elif name == 'h_pol':
            return self.pol_rnn.get_dim('states')
        elif name == 'c_pol':
            return self.pol_rnn.get_dim('cells')
        elif name == 'h_enc':
            return self.enc_rnn.get_dim('states')
        elif name == 'c_enc':
            return self.enc_rnn.get_dim('cells')
        elif name == 'h_dec':
            return self.dec_rnn.get_dim('states')
        elif name == 'c_dec':
            return self.dec_rnn.get_dim('cells')
        elif name == 'z_gen':
            return self.enc_mlp_out.get_dim('output')
        elif name in ['nll', 'kl_q2p', 'kl_p2q']:
            return 0
        else:
            super(RLDrawModel, self).get_dim(name)
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

    def set_rnn_noise(self, rnn_noise=0.0):
        """
        Set the standard deviation of "regularizing noise".
        """
        zero_ary = numpy.zeros((1,))
        new_val = zero_ary + rnn_noise
        self.rnn_noise.set_value(to_fX(new_val))
        return

    #------------------------------------------------------------------------

    @recurrent(sequences=['u'], contexts=['x'],
               states=['c', 'h_pol', 'c_pol', 'h_enc', 'c_enc', 'h_dec', 'c_dec', 'nll', 'kl_q2p', 'kl_p2q'],
               outputs=['c', 'h_pol', 'c_pol', 'h_enc', 'c_enc', 'h_dec', 'c_dec', 'nll', 'kl_q2p', 'kl_p2q'])
    def apply(self, u, c, h_pol, c_pol, h_enc, c_enc, h_dec, c_dec, nll, kl_q2p, kl_p2q, x):
        # get current state of the x under construction
        if self.step_type == 'add':
            c = c
        else:
            c = self.writer_mlp.apply(h_dec)
        c_as_x = tensor.nnet.sigmoid(tanh_clip(c, clip_val=15.0))

        # update the primary policy state
        pol_inp = tensor.concatenate([h_dec], axis=1)
        i_pol = self.pol_mlp_in.apply(pol_inp)
        h_pol, c_pol = self.pol_rnn.apply(states=h_pol, cells=c_pol,
                                          inputs=i_pol, iterate=False)

        # update the guide policy state
        enc_inp = tensor.concatenate([x, h_dec], axis=1)
        i_enc = self.enc_mlp_in.apply(enc_inp)
        h_enc, c_enc = self.enc_rnn.apply(states=h_enc, cells=c_enc,
                                          inputs=i_enc, iterate=False)

        # estimate primary policy's conditional over z
        p_z_mean, p_z_logvar, p_z = self.pol_mlp_out.apply(h_pol, u)
        if not self.use_pol:
            # use a fixed policy for "action selection" (policy is ZMUV Gauss)
            p_z_mean = 0.0 * p_z_mean
            p_z_logvar = 0.0 * p_z_logvar
            p_z = u
        # estimate guide policy's conditional over z
        q_z_mean, q_z_logvar, q_z = self.enc_mlp_out.apply(h_enc, u)

        # mix samples from p/q based on value of self.train_switch
        z = (self.train_switch[0] * q_z) + \
            ((1.0 - self.train_switch[0]) * p_z)

        # update the shared dynamics' state
        dec_inp = tensor.concatenate([z], axis=1)
        i_dec = self.dec_mlp_in.apply(dec_inp)
        h_dec, c_dec = self.dec_rnn.apply(states=h_dec, cells=c_dec, \
                                          inputs=i_dec, iterate=False)

        # get current state of the x under construction
        if self.step_type == 'add':
            c = c + self.writer_mlp.apply(h_dec)
        else:
            c = self.writer_mlp.apply(h_dec)
        # compute the NLL of the reconstruction as of this step
        c_as_x = tensor.nnet.sigmoid(tanh_clip(c, clip_val=15.0))
        nll = -1.0 * tensor.flatten(log_prob_bernoulli(x, c_as_x))
        # compute KL(q || p) and KL(p || q) for this step
        kl_q2p = tensor.sum(gaussian_kld(q_z_mean, q_z_logvar, \
                            p_z_mean, p_z_logvar), axis=1)
        kl_p2q = tensor.sum(gaussian_kld(p_gen_mean, p_gen_logvar, \
                            q_z_mean, q_z_logvar), axis=1)
        return c, h_pol, c_pol, h_enc, c_enc, h_dec, c_dec, nll, kl_q2p, kl_p2q


    #------------------------------------------------------------------------

    @application(inputs=['x_in'],
                 outputs=['cs', 'nlls', 'kl_q2ps', 'kl_p2qs'])
    def run_model(self, x_in):
        # get important size and shape information
        batch_size = x_in.shape[0]
        z_gen_dim = self.get_dim('z_gen')
        cp_dim = self.get_dim('c_pol')
        ce_dim = self.get_dim('c_enc')
        cd_dim = self.get_dim('c_dec')
        hp_dim = self.get_dim('h_pol')
        he_dim = self.get_dim('h_enc')
        hd_dim = self.get_dim('h_dec')

        # get initial states for all model components
        c0 = self.c_0.repeat(batch_size, axis=0)
        cp0 = self.cp_0.repeat(batch_size, axis=0)
        hp0 = self.hp_0.repeat(batch_size, axis=0)
        ce0 = self.ce_0.repeat(batch_size, axis=0)
        he0 = self.he_0.repeat(batch_size, axis=0)
        cd0 = self.cd_0.repeat(batch_size, axis=0)
        hd0 = self.hd_0.repeat(batch_size, axis=0)

        # get zero-mean, unit-std. Gaussian noise for use in scan op
        u_gen = self.theano_rng.normal(
                    size=(self.n_iter, batch_size, z_gen_dim),
                    avg=0., std=1.)

        # run the multi-stage guided generative process
        cs, _, _, _, _, _, _, nlls, kl_q2ps, kl_p2qs = \
                self.apply(u=u_gen, c=c0,
                           h_pol=hp0, c_pol=cp0,
                           h_enc=he0, c_enc=ce0,
                           h_dec=hd0, c_dec=cd0, x=x_in)

        # add name tags to the constructed symbolic variables
        cs.name = "cs"
        nlls.name = "nlls"
        kl_q2ps.name = "kl_q2ps"
        kl_p2qs.name = "kl_p2qs"
        return cs, nlls, kl_q2ps, kl_p2qs



    def build_model_funcs(self):
        """
        Build the symbolic costs and theano functions relevant to this model.
        """
        # symbolic variable for providing inputs
        x_in_sym = tensor.matrix('x_in_sym')

        # collect symbolic vars for model samples and costs (given x_in_sym)
        cs, nlls, kl_q2ps, kl_p2qs = self.run_model(x_in_sym)

        # get the expected NLL part of the VFE bound
        self.nll_term = tensor.mean(nlls[-1])
        self.nll_term.name = "nll_term"

        # get KL(q || p) and KL(p || q)
        self.kld_q2p_term = kl_q2ps.sum(axis=0).mean()
        self.kld_q2p_term.name = "kld_q2p_term"
        self.kld_p2q_term = kl_p2qs.sum(axis=0).mean()
        self.kld_p2q_term.name = "kld_p2q_term"

        # construct the proper VFE bound on NLL
        self.nll_bound = self.nll_term + self.kld_q2p_term

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
        self.joint_updates, applied_updates = get_adam_updates_X( \
                params=self.joint_params, \
                grads=self.joint_grads, alpha=self.lr, \
                beta1=self.mom_1, beta2=self.mom_2, \
                mom2_init=1e-3, smoothing=1e-4, max_grad_norm=10.0)

        # get the total grad norm and (post ADAM scaling) update norm.
        self.grad_norm = sum([tensor.sum(g**2.0) for g in grad_list])
        self.update_norm = sum([tensor.sum(u**2.0) for u in applied_updates])

        # collect the outputs to return from this function
        train_outputs = [self.joint_cost, self.nll_bound,
                         self.nll_term, self.kld_q2p_term, self.kld_p2q_term,
                         self.reg_term, self.grad_norm, self.update_norm]
        bound_outputs = [self.joint_cost, self.nll_bound,
                         self.nll_term, self.kld_q2p_term, self.kld_p2q_term,
                         self.reg_term]
        # collect the required inputs
        inputs = [x_in_sym]

        # compile the theano functions for computing stuff, like for real
        print("Compiling model training/update function...")
        self.train_joint = theano.function(inputs=inputs, \
                                           outputs=train_outputs, \
                                           updates=self.joint_updates)
        print("Compiling model cost estimator function...")
        self.compute_nll_bound = theano.function(inputs=inputs, \
                                                 outputs=bound_outputs)
        return

    def build_sampling_funcs(self):
        """
        Build functions for visualizing the behavior of this model.
        """
        # symbolic variable for providing inputs
        x_in_sym = tensor.matrix('x_in_sym')
        # collect symbolic vars for model samples and costs (given x_in_sym)
        cs, nlls, kl_q2ps, kl_p2qs = self.run_model(x_in_sym)
        cs_as_xs = tensor.nnet.sigmoid(tanh_clip(cs, clip_val=15.0))

        # get important parts of the VFE bound
        nll_term = nlls.mean()
        kl_term = kl_q2ps.mean()
        # grab handle for the computation graph for this model's cost
        dummy_cost = nll_term + kl_term
        self.cg = ComputationGraph([dummy_cost])

        # build the function for computing the attention trajectories
        print("Compiling model sampler...")
        sample_func = theano.function(inputs=[x_in_sym], outputs=cs_as_xs)
        def switchy_sampler(x=None, sample_source='q'):
            assert (not (x is None)), "input x is required, sorry"
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
            samps = sample_func(x)
            # set sample source switch back to previous value
            self.train_switch.set_value(old_switch)
            return samps
        self.sample_model = switchy_sampler
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
