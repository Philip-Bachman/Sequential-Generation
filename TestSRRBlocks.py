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
from SRRBlocks import SRR_LSTM
from DKCode import get_adam_updates, get_adadelta_updates
from load_data import load_udm, load_tfd, load_svhn_gray, load_binarized_mnist
from HelperFuncs import construct_masked_data, shift_and_scale_into_01, \
                        row_shuffle, to_fX

###########################
###########################
## TEST MNIST IMPUTATION ##
###########################
###########################

def test_srr_blocks():
    ##########################
    # Get some training data #
    ##########################
    rng = np.random.RandomState(1234)
    dataset = 'data/mnist.pkl.gz'
    datasets = load_udm(dataset, as_shared=False, zero_mean=False)
    Xtr = datasets[0][0]
    Xva = datasets[1][0]
    Xte = datasets[2][0]
    # Merge validation set and training set, and test on test set.
    #Xtr = np.concatenate((Xtr, Xva), axis=0)
    #Xva = Xte
    Xtr = to_fX(shift_and_scale_into_01(Xtr))
    Xva = to_fX(shift_and_scale_into_01(Xva))
    tr_samples = Xtr.shape[0]
    va_samples = Xva.shape[0]
    batch_size = 200

    ############################################################
    # Setup some parameters for the Iterative Refinement Model #
    ############################################################
    x_dim = Xtr.shape[1]
    write_dim = 300
    enc_dim = 300
    dec_dim = 300
    mix_dim = 20
    z_dim = 100
    n_iter = 20
    
    rnninits = {
        'weights_init': IsotropicGaussian(0.01),
        'biases_init': Constant(0.),
    }
    inits = {
        'weights_init': IsotropicGaussian(0.01),
        'biases_init': Constant(0.),
    }

    # setup the reader and writer (shared by primary and guide policies)
    read_dim = 2*x_dim # dimension of output from reader_mlp
    reader_mlp = Reader(x_dim=x_dim, dec_dim=dec_dim, **inits)
    writer_mlp = MLP([None, None], [dec_dim, write_dim, x_dim], \
                     name="writer_mlp", **inits)
    
    # mlps for setting conditionals over z_mix
    mix_var_mlp = CondNet([Tanh()], [x_dim, 250, mix_dim], \
                          name="mix_var_mlp", **inits)
    mix_enc_mlp = CondNet([Tanh()], [x_dim, 250, mix_dim], \
                          name="mix_enc_mlp", **inits)
    # mlp for decoding z_mix into a distribution over initial LSTM states
    mix_dec_mlp = MLP([Tanh(), Tanh()], \
                      [mix_dim, 250, (2*enc_dim + 2*dec_dim + 2*enc_dim)], \
                      name="mix_dec_mlp", **inits)
    # mlps for processing inputs to LSTMs
    var_mlp_in = MLP([Identity()], [(read_dim + dec_dim), 4*enc_dim], \
                     name="var_mlp_in", **inits)
    enc_mlp_in = MLP([Identity()], [(read_dim + dec_dim), 4*enc_dim], \
                     name="enc_mlp_in", **inits)
    dec_mlp_in = MLP([Identity()], [               z_dim, 4*dec_dim], \
                     name="dec_mlp_in", **inits)
    # mlps for turning LSTM outputs into conditionals over z_gen
    var_mlp_out = CondNet([], [enc_dim, z_dim], name="var_mlp_out", **inits)
    enc_mlp_out = CondNet([], [enc_dim, z_dim], name="enc_mlp_out", **inits)
    # LSTMs for the actual LSTMs (obviously, perhaps)
    var_rnn = BiasedLSTM(dim=enc_dim, ig_bias=2.0, fg_bias=2.0, \
                         name="var_rnn", **rnninits)
    enc_rnn = BiasedLSTM(dim=enc_dim, ig_bias=2.0, fg_bias=2.0, \
                         name="enc_rnn", **rnninits)
    dec_rnn = BiasedLSTM(dim=dec_dim, ig_bias=2.0, fg_bias=2.0, \
                         name="dec_rnn", **rnninits)

    # Construct some revelation masks
    p_masks = np.zeros((20,x_dim))
    p_masks[10] = npr.uniform(size=(1,x_dim)) < 0.25
    p_masks[-1] = np.ones((1,x_dim))
    p_masks = p_masks.astype(theano.config.floatX)
    q_masks = np.ones(p_masks.shape).astype(theano.config.floatX)
    rev_masks = [p_masks, q_masks]

    SRRM = SRR_LSTM(
                rev_masks=rev_masks,
                reader_mlp=reader_mlp,
                writer_mlp=writer_mlp,
                mix_enc_mlp=mix_enc_mlp,
                mix_dec_mlp=mix_dec_mlp,
                mix_var_mlp=mix_var_mlp,
                enc_mlp_in=enc_mlp_in,
                enc_mlp_out=enc_mlp_out,
                enc_rnn=enc_rnn,
                dec_mlp_in=dec_mlp_in,
                dec_rnn=dec_rnn,
                var_mlp_in=var_mlp_in,
                var_mlp_out=var_mlp_out,
                var_rnn=var_rnn)
    SRRM.initialize()

    # build the cost gradients, training function, samplers, etc.
    SRRM.build_model_funcs()

    #SRRM.load_model_params(f_name="SRRM_params.pkl")

    ################################################################
    # Apply some updates, to check that they aren't totally broken #
    ################################################################
    print("Beginning to train the model...")
    out_file = open("SRRM_results.txt", 'wb')
    out_file.flush()
    costs = [0. for i in range(10)]
    learn_rate = 0.0002
    momentum = 0.75
    batch_idx = np.arange(batch_size) + tr_samples
    for i in range(250000):
        scale = min(1.0, ((i+1) / 1000.0))
        if (((i + 1) % 10000) == 0):
            learn_rate = learn_rate * 0.95
        if (i > 10000):
            momentum = 0.95
        else:
            momentum = 0.75
        # get the indices of training samples for this batch update
        batch_idx += batch_size
        if (np.max(batch_idx) >= tr_samples):
            # we finished an "epoch", so we rejumble the training set
            Xtr = row_shuffle(Xtr)
            batch_idx = np.arange(batch_size)
        # set sgd and objective function hyperparams for this update
        zero_ary = np.zeros((1,))
        SRRM.lr.set_value(to_fX(zero_ary + learn_rate))
        SRRM.mom_1.set_value(to_fX(zero_ary + momentum))
        SRRM.mom_2.set_value(to_fX(zero_ary + 0.99))

        # perform a minibatch update and record the cost for this batch
        Xb = to_fX(Xtr.take(batch_idx, axis=0))
        result = SRRM.train_joint(Xb)

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
            SRRM.save_model_params("SRRM_params.pkl")
            # compute a small-sample estimate of NLL bound on validation set
            Xva = row_shuffle(Xva)
            Xb = to_fX(Xva[:5000])
            va_costs = SRRM.compute_nll_bound(Xb)
            str1 = "    va_nll_bound : {}".format(va_costs[1])
            str2 = "    va_nll_term  : {}".format(va_costs[2])
            str3 = "    va_kld_q2p   : {}".format(va_costs[3])
            joint_str = "\n".join([str1, str2, str3])
            print(joint_str)
            out_file.write(joint_str+"\n")
            out_file.flush()

if __name__=="__main__":
    test_srr_blocks()
