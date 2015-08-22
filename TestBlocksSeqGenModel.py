##################################################################
# Code for testing the variational Multi-Stage Generative Model. #
##################################################################

from __future__ import print_function, division

# basic python
import cPickle as pickle
from PIL import Image
import numpy as np
import numpy.random as npr
from collections import OrderedDict

# theano business
import theano
import theano.tensor as T

# blocks stuff
from blocks.initialization import Constant, IsotropicGaussian, Orthogonal
from blocks.filter import VariableFilter
from blocks.graph import ComputationGraph
from blocks.roles import PARAMETER
from blocks.model import Model
from blocks.bricks import Tanh, Identity, Rectifier
from blocks.bricks.cost import BinaryCrossEntropy
from blocks.bricks.recurrent import SimpleRecurrent, LSTM

# phil's sweetness
import utils
from BlocksModels import *
from DKCode import get_adam_updates, get_adadelta_updates
from load_data import load_binarized_mnist
from HelperFuncs import construct_masked_data, shift_and_scale_into_01, \
                        row_shuffle, to_fX

###################################
###################################
## HELPER FUNCTIONS FOR SAMPLING ##
###################################
###################################

def scale_norm(arr):
    arr = arr - arr.min()
    scale = (arr.max() - arr.min())
    return scale * arr

def img_grid(arr, global_scale=True):
    N, height, width = arr.shape

    rows = int(np.sqrt(N))
    cols = int(np.sqrt(N))

    if rows*cols < N:
        cols = cols + 1

    if rows*cols < N:
        rows = rows + 1

    total_height = rows * height
    total_width  = cols * width

    if global_scale:
        arr = scale_norm(arr)

    I = np.zeros((total_height, total_width))

    for i in xrange(N):
        r = i // cols
        c = i % cols

        if global_scale:
            this = arr[i]
        else:
            this = scale_norm(arr[i])

        offset_y, offset_x = r*height, c*width
        I[offset_y:(offset_y+height), offset_x:(offset_x+width)] = this
    
    I = (255*I).astype(np.uint8)
    return Image.fromarray(I)

###########################
###########################
## TEST MNIST IMPUTATION ##
###########################
###########################

def test_sgm_mnist(step_type='add', occ_dim=14, drop_prob=0.0, attention=False):
    ##########################
    # Get some training data #
    ##########################
    rng = np.random.RandomState(1234)
    Xtr, Xva, Xte = load_binarized_mnist(data_path='./data/')
    Xtr = np.vstack((Xtr, Xva))
    Xva = Xte
    #del Xte
    tr_samples = Xtr.shape[0]
    va_samples = Xva.shape[0]
    batch_size = 200

    ############################################################
    # Setup some parameters for the Iterative Refinement Model #
    ############################################################
    x_dim = Xtr.shape[1]
    writer_dim = 250
    reader_dim = 250
    dyn_dim = 250
    primary_dim = 500
    guide_dim = 500
    z_dim = 100
    n_iter = 20
    dp_int = int(100.0 * drop_prob)
    
    rnninits = {
        'weights_init': IsotropicGaussian(0.01),
        'biases_init': Constant(0.),
    }
    inits = {
        'weights_init': IsotropicGaussian(0.01),
        'biases_init': Constant(0.),
    }

    att_tag = "NA" # attention not implemented yet

    # reader MLP provides input to the dynamics LSTM update
    reader_mlp = MLP([Rectifier(), Rectifier(), None], \
                     [(x_dim + z_dim), reader_dim, reader_dim, 4*dyn_dim], \
                     name="reader_mlp", **inits)
    # writer MLP applies changes to the generation workspace
    writer_mlp = MLP([Rectifier(), Rectifier(), None], \
                     [(dyn_dim + z_dim), writer_dim, writer_dim, x_dim], \
                     name="writer_mlp", **inits)

    # MLPs for computing conditionals over z
    primary_policy = CondNet([Rectifier(), Rectifier()], \
                             [(dyn_dim + x_dim), primary_dim, primary_dim, z_dim], \
                             name="primary_policy", **inits)
    guide_policy = CondNet([Rectifier(), Rectifier()], \
                           [(dyn_dim + 2*x_dim), guide_dim, guide_dim, z_dim], \
                           name="guide_policy", **inits)
    # LSTMs for the actual LSTMs (obviously, perhaps)
    shared_dynamics = BiasedLSTM(dim=dyn_dim, ig_bias=2.0, fg_bias=2.0, \
                                 name="shared_dynamics", **rnninits)

    model = SeqGenModel(
                n_iter,
                step_type=step_type, # step_type can be 'add' or 'jump'
                reader_mlp=reader_mlp,
                writer_mlp=writer_mlp,
                primary_policy=primary_policy,
                guide_policy=guide_policy,
                shared_dynamics=shared_dynamics)
    model.initialize()

    # build the cost gradients, training function, samplers, etc.
    model.build_model_funcs()

    #model.load_model_params(f_name="TBSGM_IMP_MNIST_PARAMS_OD{}_DP{}_{}_{}.pkl".format(occ_dim, dp_int, step_type, att_tag))

    ################################################################
    # Apply some updates, to check that they aren't totally broken #
    ################################################################
    print("Beginning to train the model...")
    out_file = open("TBSGM_IMP_MNIST_RESULTS_OD{}_DP{}_{}_{}.txt".format(occ_dim, dp_int, step_type, att_tag), 'wb')
    out_file.flush()
    costs = [0. for i in range(10)]
    learn_rate = 0.0002
    momentum = 0.5
    batch_idx = np.arange(batch_size) + tr_samples
    for i in range(250000):
        scale = min(1.0, ((i+1) / 1000.0))
        if (((i + 1) % 10000) == 0):
            learn_rate = learn_rate * 0.95
        if (i > 10000):
            momentum = 0.90
        else:
            momentum = 0.50
        # get the indices of training samples for this batch update
        batch_idx += batch_size
        if (np.max(batch_idx) >= tr_samples):
            # we finished an "epoch", so we rejumble the training set
            Xtr = row_shuffle(Xtr)
            batch_idx = np.arange(batch_size)
        # set sgd and objective function hyperparams for this update
        zero_ary = np.zeros((1,))
        model.lr.set_value(to_fX(zero_ary + learn_rate))
        model.mom_1.set_value(to_fX(zero_ary + momentum))
        model.mom_2.set_value(to_fX(zero_ary + 0.99))

        # perform a minibatch update and record the cost for this batch
        Xb = to_fX(Xtr.take(batch_idx, axis=0))
        _, Xb, Mb = construct_masked_data(Xb, drop_prob=drop_prob, \
                                    occ_dim=occ_dim, data_mean=None)
        result = model.train_joint(Xb, Mb)

        costs = [(costs[j] + result[j]) for j in range(len(result))]
        if ((i % 200) == 0):
            costs = [(v / 200.0) for v in costs]
            str1 = "-- batch {0:d} --".format(i)
            str2 = "    total_cost: {0:.4f}".format(costs[0])
            str3 = "    nll_bound : {0:.4f}".format(costs[1])
            str4 = "    nll_term  : {0:.4f}".format(costs[2])
            str5 = "    kld_q2p   : {0:.4f}".format(costs[3])
            str6 = "    kld_p2q   : {0:.4f}".format(costs[4])
            str7 = "    reg_term  : {0:.4f}".format(costs[5])
            joint_str = "\n".join([str1, str2, str3, str4, str5, str6, str7])
            print(joint_str)
            out_file.write(joint_str+"\n")
            out_file.flush()
            costs = [0.0 for v in costs]
        if ((i % 1000) == 0):
            model.save_model_params("TBSGM_IMP_MNIST_PARAMS_OD{}_DP{}_{}_{}.pkl".format(occ_dim, dp_int, step_type, att_tag))
            # compute a small-sample estimate of NLL bound on validation set
            Xva = row_shuffle(Xva)
            Xb = to_fX(Xva[:5000])
            _, Xb, Mb = construct_masked_data(Xb, drop_prob=drop_prob, \
                                    occ_dim=occ_dim, data_mean=None)
            va_costs = model.compute_nll_bound(Xb, Mb)
            str1 = "    va_nll_bound : {}".format(va_costs[1])
            str2 = "    va_nll_term  : {}".format(va_costs[2])
            str3 = "    va_kld_q2p   : {}".format(va_costs[3])
            joint_str = "\n".join([str1, str2, str3])
            print(joint_str)
            out_file.write(joint_str+"\n")
            out_file.flush()
            # draw some independent samples from the model
            Xb = to_fX(Xva[:100])
            _, Xb, Mb = construct_masked_data(Xb, drop_prob=drop_prob, \
                                    occ_dim=occ_dim, data_mean=None)
            samples, _ = model.do_sample(Xb, Mb)
            n_iter, N, D = samples.shape
            samples = samples.reshape( (n_iter, N, 28, 28) )
            for j in xrange(n_iter):
                img = img_grid(samples[j,:,:,:])
                img.save("TBSGM-IMP-MNIST-OD{0:d}-DP{1:d}-{2:s}-samples-{3:03d}.png".format(occ_dim, dp_int, step_type, j))

if __name__=="__main__":
    test_sgm_mnist(step_type='add', occ_dim=0, drop_prob=0.85)