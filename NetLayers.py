import numpy as np
import numpy.random as npr
import theano
import theano.tensor as T
#from theano.tensor.shared_randomstreams import RandomStreams as RandStream
from theano.sandbox.cuda.rng_curand import CURAND_RandomStreams as RandStream
from theano.sandbox.cuda.dnn import dnn_conv, dnn_pool


###############################
# ACTIVATIONS AND OTHER STUFF #
###############################

def row_normalize(x):
    """Normalize rows of matrix x to unit (L2) norm."""
    x_normed = x / T.sqrt(T.sum(x**2.,axis=1,keepdims=1) +  constFX(1e-8))
    return x_normed

def col_normalize(x):
    """Normalize cols of matrix x to unit (L2) norm."""
    x_normed = x / T.sqrt(T.sum(x**2.,axis=0,keepdims=1) + constFX(1e-8))
    return x_normed

def max_normalize(x, axis=1):
    """
    Normalize matrix x to max unit (L2) norm along some axis.
    """
    row_norms = T.sqrt(T.sum(x**2., axis=axis, keepdims=1))
    row_scales = T.maximum(1.0, row_norms)
    x_normed = x / row_scales
    return x_normed

def rehu_actfun(x):
    """Compute rectified huberized activation for x."""
    M_quad = (x > 0.0) * (x < 0.5)
    M_line = (x >= 0.5)
    x_rehu = (M_quad * x**2.) + (M_line * (x - 0.25))
    return x_rehu

def relu_actfun(x):
    """Compute rectified linear activation for x."""
    x_relu = T.maximum(0., x)
    return x_relu

def softplus_actfun(x, scale=None):
    """Compute rescaled softplus activation for x."""
    if scale is None:
        x_softplus = (1.0 / 2.0) * T.nnet.softplus(2.0*x)
    else:
        x_softplus = (1.0 / scale) * T.nnet.softplus(scale*x)
    return x_softplus

def tanh_actfun(x, scale=None):
    """Compute  rescaled tanh activation for x."""
    if scale is None:
        x_tanh = T.tanh(x)
    else:
        x_tanh = scale * T.tanh(constFX(1/scale) * x)
    return x_tanh

def maxout_actfun(input, pool_size, filt_count):
    """Apply maxout over non-overlapping sets of values."""
    last_start = filt_count - pool_size
    mp_vals = None
    for i in xrange(pool_size):
        cur = input[:,i:(last_start+i+1):pool_size]
        if mp_vals is None:
            mp_vals = cur
        else:
            mp_vals = T.maximum(mp_vals, cur)
    return mp_vals

def normout_actfun(input, pool_size, filt_count):
    """Apply (L2) normout over non-overlapping sets of values."""
    l_start = filt_count - pool_size
    relu_vals = T.stack(\
        *[input[:,i:(l_start+i+1):pool_size] for i in range(pool_size)])
    pooled_vals = T.sqrt(T.mean(relu_vals**2.0, axis=0))
    return pooled_vals

def noop_actfun(x):
    """Do nothing activation. For output layer probably."""
    return x

def safe_softmax(x):
    """Softmax that shouldn't overflow."""
    e_x = T.exp(x - T.max(x, axis=1, keepdims=True))
    x_sm = e_x / T.sum(e_x, axis=1, keepdims=True)
    #x_sm = T.nnet.softmax(x)
    return x_sm

def smooth_softmax(x):
    """Softmax that shouldn't overflow, with Laplacish smoothing."""
    eps = 0.0001
    e_x = T.exp(x - T.max(x, axis=1, keepdims=True))
    p = (e_x / T.sum(e_x, axis=1, keepdims=True)) + constFX(eps)
    p_sm = p / T.sum(p, axis=1, keepdims=True)
    return p_sm

def smooth_kl_divergence(p, q):
    """Measure the KL-divergence from "approximate" distribution q to "true"
    distribution p. Use smoothed softmax to convert p and q from encodings
    in terms of relative log-likelihoods into sum-to-one distributions."""
    p_sm = smooth_softmax(p)
    q_sm = smooth_softmax(q)
    # This term is: cross_entropy(p, q) - entropy(p)
    kl_sm = T.sum(((T.log(p_sm) - T.log(q_sm)) * p_sm), axis=1, keepdims=True)
    return kl_sm

def smooth_cross_entropy(p, q):
    """Measure the cross-entropy between "approximate" distribution q and
    "true" distribution p. Use smoothed softmax to convert p and q from
    encodings in terms of relative log-likelihoods into sum-to-one dists."""
    p_sm = smooth_softmax(p)
    q_sm = smooth_softmax(q)
    # This term is: entropy(p) + kl_divergence(p, q)
    ce_sm = -T.sum((p_sm * T.log(q_sm)), axis=1, keepdims=True)
    return ce_sm

def apply_mask(Xd=None, Xc=None, Xm=None):
    """
    Apply a mask, like in the old days.
    """
    X_masked = ((1.0 - Xm) * Xd) + (Xm * Xc)
    return X_masked

def binarize_data(X):
    """
    Make a sample of bernoulli variables with probabilities given by X.
    """
    X_shape = X.shape
    probs = npr.rand(*X_shape)
    X_binary = 1.0 * (probs < X)
    return X_binary.astype(theano.config.floatX)

def row_shuffle(X, Y=None):
    """
    Return a copy of X with shuffled rows.
    """
    shuf_idx = np.arange(X.shape[0])
    npr.shuffle(shuf_idx)
    X_shuf = X[shuf_idx]
    if Y is None:
        result = X_shuf
    else:
        Y_shuf = Y[shuf_idx]
        result = [X_shuf, Y_shuf]
    return result

# code from the lasagne library
def ortho_matrix(shape=None, gain=1.0):
    """
    Orthogonal matrix initialization. For n-dimensional shapes where n > 2,
    the n-1 trailing axes are flattened. For convolutional layers, this
    corresponds to the fan-in, so this makes the initialization usable for
    both dense and convolutional layers.
    """
    # relu gain for elided activations
    if gain == 'relu':
        gain = np.sqrt(2)
    if len(shape) < 2:
        raise RuntimeError("Only shapes of length 2 or more are supported.")
    flat_shape = (shape[0], np.prod(shape[1:]))
    a = npr.normal(0.0, 1.0, flat_shape)
    u, _, v = np.linalg.svd(a, full_matrices=False)
    q = u if u.shape == flat_shape else v # pick the one with the correct shape
    q = q.reshape(shape)
    W = gain * q[:shape[0], :shape[1]]
    return W

# Xavier Glorot initialization
def glorot_matrix(shape):
    """
    Xavier Glorot initialization -- from the AISTATS 2010 paper:
        "Understanding the Difficulty of Training Deep Feedforward Networks"
    """
    if not (len(shape) == 2):
        raise RuntimeError("Only shapes of length are supported.")
    w_scale = np.sqrt(6.0) / np.sqrt(shape[0] + shape[1])
    W = npr.uniform(low=-w_scale, high=w_scale, size=shape)
    return W

def batchnorm(X, rescale=None, reshift=None, u=None, s=None, e=1e-8):
    """
    batchnorm with support for not using scale and shift parameters
    as well as inference values (u and s) and partial batchnorm (via a)
    will detect and use convolutional or fully connected version
    """
    g = rescale
    b = reshift
    if X.ndim == 4:
        if u is not None and s is not None:
            # use normalization params given a priori
            b_u = u.dimshuffle('x', 0, 'x', 'x')
            b_s = s.dimshuffle('x', 0, 'x', 'x')
        else:
            # compute normalization params from input
            b_u = T.mean(X, axis=[0, 2, 3]).dimshuffle('x', 0, 'x', 'x')
            b_s = T.mean(T.sqr(X - b_u), axis=[0, 2, 3]).dimshuffle('x', 0, 'x', 'x')
        # batch normalize
        X = (X - b_u) / T.sqrt(b_s + e)
        if g is not None and b is not None:
            # apply rescale and reshift
            X = X*T.exp(0.2*g.dimshuffle('x', 0, 'x', 'x')) + b.dimshuffle('x', 0, 'x', 'x')
    elif X.ndim == 2:
        if u is None and s is None:
            # compute normalization params from input
            u = T.mean(X, axis=0)
            s = T.mean(T.sqr(X - u), axis=0)
        # batch normalize
        X = (X - u) / T.sqrt(s + e)
        if g is not None and b is not None:
            # apply rescale and reshift
            X = X*T.exp(0.2*g) + b
    else:
        raise NotImplementedError
    return X

######################################
# BASIC FULLY-CONNECTED HIDDEN LAYER #
######################################

class HiddenLayer(object):
    def __init__(self, rng, input, in_dim, out_dim,
                 activation=None, pool_size=0,
                 drop_rate=0.,
                 W=None, b=None, b_in=None, s_in=None,
                 name="", W_scale=1.0, apply_bn=False):

        # Setup a shared random generator for this layer
        self.rng = RandStream(rng.randint(1000000))

        # setup parameters for controlling
        zero_ary = np.zeros((1,)).astype(theano.config.floatX)
        self.drop_rate = theano.shared(value=(zero_ary+drop_rate),
                name="{0:s}_drop_rate".format(name))
        self.apply_bn = apply_bn

        # setup scale and bias params for the input
        if b_in is None:
            # batch normalization reshifts are initialized to zero
            ary = np.zeros((out_dim,), dtype=theano.config.floatX)
            b_in = theano.shared(value=ary, name="{0:s}_b_in".format(name))
        if s_in is None:
            # batch normalization rescales are initialized to zero
            ary = np.zeros((out_dim,), dtype=theano.config.floatX)
            s_in = theano.shared(value=ary, name="{0:s}_s_in".format(name))
        self.b_in = b_in
        self.s_in = s_in

        # Set some basic layer properties
        self.pool_size = pool_size
        self.in_dim = in_dim
        self.out_dim = out_dim
        if self.pool_size <= 1:
            self.filt_count = self.out_dim
        else:
            self.filt_count = self.out_dim * self.pool_size
        self.pool_count = self.filt_count / max(self.pool_size, 1)
        if activation is None:
            activation = relu_actfun
        if self.pool_size <= 1:
            self.activation = activation
        else:
            self.activation = lambda x: \
                    maxout_actfun(x, self.pool_size, self.filt_count)

        # Get some random initial weights and biases, if not given
        if W is None:
            # Generate initial filters using orthogonal random trick
            W_shape = (self.in_dim, self.filt_count)
            if W_scale == 'xg':
                W_init = glorot_matrix(W_shape)
            else:
                #W_init = (W_scale * (1.0 / np.sqrt(self.in_dim))) * \
                #          npr.normal(0.0, 1.0, W_shape)
                W_init = ortho_matrix(shape=W_shape, gain=W_scale)
            W_init = W_init.astype(theano.config.floatX)
            W = theano.shared(value=W_init, name="{0:s}_W".format(name))
        if b is None:
            b_init = np.zeros((self.filt_count,), dtype=theano.config.floatX)
            b = theano.shared(value=b_init, name="{0:s}_b".format(name))

        # Set layer weights and biases
        self.W = W
        self.b = b

        # Feedforward through the layer
        use_drop = drop_rate > 0.001
        self.linear_output, self.noisy_linear, self.output = \
                self.apply(input, use_drop=use_drop)

        # Compute some properties of the activations, probably to regularize
        self.act_l2_sum = T.sum(self.noisy_linear**2.) / self.output.size

        # Conveniently package layer parameters
        self.params = [self.W, self.b, self.b_in, self.s_in]
        self.shared_param_dicts = {
                'W': self.W,
                'b': self.b,
                'b_in': self.b_in,
                's_in': self.s_in }
        # Layer construction complete...
        return

    def apply(self, input, use_drop=False):
        """
        Apply feedforward to this input, returning several partial results.
        """
        # Apply masking noise to the input (if desired)
        if use_drop:
            input = self._drop_from_input(input, self.drop_rate[0])
        # Compute linear "pre-activation" for this layer
        linear_output = T.dot(input, self.W) + self.b
        # Apply batch normalization if desired
        if self.apply_bn:
            linear_output = batchnorm(linear_output, rescale=self.s_in,
                                      reshift=self.b_in, u=None, s=None)
        # Apply activation function
        final_output = self.activation(linear_output)
        # package partial results for easy return
        results = [final_output, linear_output]
        return results

    def _drop_from_input(self, input, p):
        """p is the probability of dropping elements of input."""
        # get a drop mask that drops things with probability p
        drop_rnd = self.rng.uniform(size=input.shape, low=0.0, high=1.0, \
                dtype=theano.config.floatX)
        drop_mask = drop_rnd > p
        # get a scaling factor to keep expectations fixed after droppage
        drop_scale = 1. / (1. - p)
        # apply dropout mask and rescaling factor to the input
        droppy_input = drop_scale * input * drop_mask
        return droppy_input

    def _noisy_params(self, P, noise_lvl=0.):
        """Noisy weights, like convolving energy surface with a gaussian."""
        P_nz = P + self.rng.normal(size=P.shape, avg=0.0, std=noise_lvl, \
                dtype=theano.config.floatX)
        return P_nz


######################################################
# SIMPLE LAYER FOR AVERAGING OUTPUTS OF OTHER LAYERS #
######################################################

class JoinLayer(object):
    """
    Simple layer that averages over "linear_output"s of other layers.

    Note: The list of layers to average over is the only parameter used.
    """
    def __init__(self, input_layers):
        print("making join layer over {0:d} output layers...".format( \
                len(input_layers)))
        il_los = [il.linear_output for il in input_layers]
        self.output = T.mean(T.stack(*il_los), axis=0)
        self.linear_output = self.output
        self.noisy_linear_output = self.output
        return

#############################################
# RESHAPING LAYERS (FOR VECTORS<-->TENSORS) #
#############################################

class Reshape2D4DLayer(object):
	"""
	Reshape from flat vectors to image-y 3D tensors.
	"""
	def __init__(self, input=None, out_shape=None):
		assert(len(out_shape) == 3)
		self.input = input
		self.output = self.input.reshape((self.input.shape[0], \
			out_shape[0], out_shape[1], out_shape[2]))
		self.linear_output = self.output
		self.noisy_linear_output = self.output
		return

class Reshape4D2DLayer(object):
	"""
	Flatten from 3D image-y tensors to flat vectors.
	"""
	def __init__(self, input=None):
		self.input = input
		out_dim = T.prod(self.input.shape[1:])
		self.output = self.input.reshape((self.input.shape[0], out_dim))
		self.linear_output = self.output
		self.noisy_linear_output = self.output
		return

#####################################################
# DISCRIMINATIVE LAYER (SINGLE-OUTPUT LINEAR LAYER) #
#####################################################

class DiscLayer(object):
    def __init__(self, rng, input, in_dim, W=None, b=None, W_scale=1.0):
        # Setup a shared random generator for this layer
        self.rng = RandStream(rng.randint(1000000))

        self.input = input
        self.in_dim = in_dim

        # Get some random initial weights and biases, if not given
        if W is None:
            # Generate random initial filters in a typical way
            W_init = 1.0 * np.asarray(rng.normal( \
                      size=(self.in_dim, 1)), \
                      dtype=theano.config.floatX)
            W = theano.shared(value=(W_scale*W_init))
        if b is None:
            b_init = np.zeros((1,), dtype=theano.config.floatX)
            b = theano.shared(value=b_init)

        # Set layer weights and biases
        self.W = W
        self.b = b

        # Compute linear "pre-activation" for this layer
        self.linear_output = 20.0 * T.tanh((T.dot(self.input, self.W) + self.b) / 20.0)

        # Apply activation function
        self.output = self.linear_output

        # Compute squared sum of outputs, for regularization
        self.act_l2_sum = T.sum(self.output**2.0) / self.output.shape[0]

        # Conveniently package layer parameters
        self.params = [self.W, self.b]
        # little layer construction complete...
        return

    def _noisy_params(self, P, noise_lvl=0.):
        """Noisy weights, like convolving energy surface with a gaussian."""
        P_nz = P + DCG(self.rng.normal(size=P.shape, avg=0.0, std=noise_lvl, \
                dtype=theano.config.floatX))
        return P_nz
