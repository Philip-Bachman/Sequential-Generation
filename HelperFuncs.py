import time
import utils as utils
import numpy as np
import numpy.random as npr
import theano
import theano.tensor as T

##################################
# MISCELLANEOUS HELPER FUNCTIONS #
##################################

def reparametrize(z_mean, z_logvar, rng=None, rvs=None):
    """
    Gaussian reparametrization helper function.
    """
    assert not ((rng is None) and (rvs is None)), \
            "must provide either rng or rvs."
    assert ((rng is None) or (rvs is None)), \
            "must provide either rng or rvs."
    if not (rng is None):
        # generate zmuv samples from the provided rng
        zmuv_gauss = rng.normal(size=z_mean.shape, dtype=theano.config.floatX)
    else:
        # use zmuv samples provided by the user
        zmuv_gauss = rvs
    reparam_gauss = z_mean + (T.exp(0.5*z_logvar) * zmuv_gauss)
    return reparam_gauss

def tanh_clip(x, clip_val=10.0):
    x_clipped = clip_val * T.tanh((1. / clip_val) * x)
    return x_clipped

def DCG(x):
    x_dcg = theano.gradient.disconnected_grad(x)
    return x_dcg

def constFX(x):
    """Cast x as constant TensorVariable with dtype floatX."""
    x_CFX = T.constant(x, dtype=theano.config.floatX)
    return x_CFX

def to_fX(np_ary):
    np_ary_fX = np_ary.astype(theano.config.floatX)
    return np_ary_fX

def posterior_klds(IN, Xtr, batch_size, batch_count):
    """
    Get posterior KLd cost for some inputs from Xtr.
    """
    post_klds = []
    for i in range(batch_count):
        batch_idx = npr.randint(low=0, high=Xtr.shape[0], size=(batch_size,))
        X = Xtr.take(batch_idx, axis=0)
        post_klds.extend([k for k in IN.kld_func(X)])
    return post_klds

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

#####################################
# HELPER FUNCTIONS FOR DATA MASKING #
#####################################

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

def sample_masks(X, drop_prob=0.3):
    """
    Sample a binary mask to apply to the matrix X, with rate mask_prob.
    """
    probs = npr.rand(*X.shape)
    mask = 1.0 * (probs > drop_prob)
    return mask.astype(theano.config.floatX)

def sample_patch_masks(X, im_shape, patch_shape):
    """
    Sample a random patch mask for each image in X.
    """
    obs_count = X.shape[0]
    rs = patch_shape[0]
    cs = patch_shape[1]
    off_row = npr.randint(1,high=(im_shape[0]-rs-1), size=(obs_count,))
    off_col = npr.randint(1,high=(im_shape[1]-cs-1), size=(obs_count,))
    dummy = np.zeros(im_shape)
    mask = np.zeros(X.shape)
    for i in range(obs_count):
        dummy = (0.0 * dummy) + 1.0
        dummy[off_row[i]:(off_row[i]+rs), off_col[i]:(off_col[i]+cs)] = 0.0
        mask[i,:] = dummy.ravel()
    return mask.astype(theano.config.floatX)

def collect_obs_costs(batch_costs, batch_reps):
    """
    Collect per-observation costs from a cost vector containing the cost for
    multiple repetitions of each observation.
    """
    obs_count = int(batch_costs.shape[0] / batch_reps)
    obs_costs = np.zeros((obs_count,))
    obs_idx = -1
    for i in range(batch_costs.shape[0]):
        if ((i % batch_reps) == 0):
            obs_idx = obs_idx + 1
        obs_costs[obs_idx] = obs_costs[obs_idx] + batch_costs[i]
    obs_costs = obs_costs / batch_reps
    return obs_costs

def construct_masked_data(xi, \
                          drop_prob=0.0, \
                          occ_dim=None, \
                          data_mean=None):
    """
    Construct randomly masked data from xi.
    """
    if data_mean is None:
        data_mean = np.zeros((xi.shape[1],))
    im_dim = int(xi.shape[1]**0.5) # images should be square
    xo = xi.copy()
    if drop_prob > 0.0:
        # apply fully-random occlusion
        xm_rand = sample_masks(xi, drop_prob=drop_prob)
    else:
        # don't apply fully-random occlusion
        xm_rand = np.ones(xi.shape)
    if occ_dim is None:
        # don't apply rectangular occlusion
        xm_patch = np.ones(xi.shape)
    else:
        # apply rectangular occlusion
        xm_patch = sample_patch_masks(xi, (im_dim,im_dim), (occ_dim,occ_dim))
    xm = xm_rand * xm_patch
    xi = (xm * xi) + ((1.0 - xm) * data_mean)
    xi = to_fX(xi)
    xo = to_fX(xo)
    xm = to_fX(xm)
    return xi, xo, xm

def sample_data_masks(xi, drop_prob=0.0, occ_dim=None):
    """
    Construct random masks for observations in xi.
    """
    im_dim = int(xi.shape[1]**0.5) # images should be square
    if drop_prob > 0.0:
        # apply fully-random occlusion
        xm_rand = sample_masks(xi, drop_prob=drop_prob)
    else:
        # don't apply fully-random occlusion
        xm_rand = np.ones(xi.shape)
    if occ_dim is None:
        # don't apply rectangular occlusion
        xm_patch = np.ones(xi.shape)
    else:
        # apply rectangular occlusion
        xm_patch = sample_patch_masks(xi, (im_dim,im_dim), (occ_dim,occ_dim))
    xm = xm_rand * xm_patch
    xm = to_fX(xm)
    return xm

def shift_and_scale_into_01(X):
    X = X - np.min(X, axis=1, keepdims=True)
    X = X / np.max(X, axis=1, keepdims=True)
    return X


def one_hot(Yc, cat_dim=None):
    """
    Given a tensor Yc of dimension d with integer values from range(cat_dim),
    return new tensor of dimension d + 1 with values 0/1, where the last
    dimension gives a one-hot representation of the values in Yc.

    if cat_dim is not given, cat_dim is set to max(Yc) + 1
    """
    if cat_dim is None:
        cat_dim = T.max(Yc) + 1
    ranges = T.shape_padleft(T.arange(cat_dim), Yc.ndim)
    Yoh = T.eq(ranges, T.shape_padright(Yc, 1))
    return Yoh

def one_hot_np(Yc, cat_dim=None):
    """
    Given a numpy integer column vector Yc, generate a matrix Yoh in which
    Yoh[i,:] is a one-hot vector -- Yoh[i,Yc[i]] = 1.0 and other Yoh[i,j] = 0
    """
    if cat_dim is None:
        cat_dim = np.max(Yc) + 1
    Yoh = np.zeros((Yc.size, cat_dim))
    Yoh[np.arange(Yc.size),Yc.flatten()] = 1.0
    return Yoh
