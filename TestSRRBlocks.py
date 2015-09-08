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
from SRRBlocks import ImgScan
from DKCode import get_adam_updates, get_adadelta_updates
from load_data import load_udm, load_tfd, load_svhn_gray, load_binarized_mnist
from HelperFuncs import construct_masked_data, shift_and_scale_into_01, \
                        row_shuffle, to_fX

################################
################################
## TEST COLUMN-WISE GENERATOR ##
################################
################################

def test_img_scan(attention=False):
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
    x_dim = 28
    im_shape = (28,28)
    inner_steps = 5
    rnn_dim = 128
    write_dim = 64
    mlp_dim = 128
    z_dim = 50

    rnninits = {
        'weights_init': IsotropicGaussian(0.01),
        'biases_init': Constant(0.),
    }
    inits = {
        'weights_init': IsotropicGaussian(0.01),
        'biases_init': Constant(0.),
    }

    # setup the reader and writer
    if attention:
        read_N, write_N = (2, 5) # resolution of reader and writer
        read_dim = 2*read_N**2   # total number of "pixels" read by reader
        reader_mlp = AttentionReader(x_dim=x_dim, dec_dim=dec_dim,
                                 width=28, height=28,
                                 N=read_N, **inits)
        writer_mlp = AttentionWriter(input_dim=dec_dim, output_dim=x_dim,
                                 width=28, height=28,
                                 N=write_N, **inits)
        att_tag = "YA"
    else:
        read_dim = 2*x_dim
        reader_mlp = Reader(x_dim=x_dim, dec_dim=dec_dim, **inits)
        writer_mlp = MLP([None, None], [dec_dim, write_dim, x_dim], \
                         name="writer_mlp", **inits)
        att_tag = "NA"

    # setup the reader and writer (shared by primary and guide policies)
    read_dim = 2*x_dim # dimension of output from reader_mlp
    reader_mlp = Reader(x_dim=x_dim, dec_dim=rnn_dim, **inits)
    writer_mlp = MLP([None, None], [rnn_dim, write_dim, x_dim], \
                     name="writer_mlp", **inits)

    # mlps for processing inputs to LSTMs
    con_mlp_in = MLP([Identity()], [                       z_dim, 4*rnn_dim], \
                     name="con_mlp_in", **inits)
    var_mlp_in = MLP([Identity()], [(x_dim + read_dim + rnn_dim), 4*rnn_dim], \
                     name="var_mlp_in", **inits)
    gen_mlp_in = MLP([Identity()], [(x_dim + read_dim + rnn_dim), 4*rnn_dim], \
                     name="gen_mlp_in", **inits)

    # mlps for turning LSTM outputs into conditionals over z_gen
    gen_mlp_out = CondNet([], [rnn_dim, z_dim], name="gen_mlp_out", **inits)
    var_mlp_out = CondNet([], [rnn_dim, z_dim], name="var_mlp_out", **inits)
    # LSTMs for the actual LSTMs (obviously, perhaps)
    con_rnn = BiasedLSTM(dim=rnn_dim, ig_bias=2.0, fg_bias=2.0, \
                         name="con_rnn", **rnninits)
    gen_rnn = BiasedLSTM(dim=rnn_dim, ig_bias=2.0, fg_bias=2.0, \
                         name="gen_rnn", **rnninits)
    var_rnn = BiasedLSTM(dim=rnn_dim, ig_bias=2.0, fg_bias=2.0, \
                         name="var_rnn", **rnninits)

    ImgScan_doc_str = \
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

    IMS = ImgScan(
                im_shape=im_shape,
                inner_steps=inner_steps,
                reader_mlp=reader_mlp,
                writer_mlp=writer_mlp,
                con_mlp_in=con_mlp_in,
                con_rnn=con_rnn,
                gen_mlp_in=gen_mlp_in,
                gen_mlp_out=gen_mlp_out,
                gen_rnn=gen_rnn,
                var_mlp_in=var_mlp_in,
                var_mlp_out=var_mlp_out,
                var_rnn=var_rnn)
    IMS.initialize()

    # build the cost gradients, training function, samplers, etc.
    IMS.build_model_funcs()

    #IMS.load_model_params(f_name="SRRM_params.pkl")

    ################################################################
    # Apply some updates, to check that they aren't totally broken #
    ################################################################
    print("Beginning to train the model...")
    out_file = open("IMS_results.txt", 'wb')
    out_file.flush()
    costs = [0. for i in range(10)]
    learn_rate = 0.0002
    momentum = 0.75
    batch_idx = np.arange(batch_size) + tr_samples
    for i in range(250000):
        scale = min(1.0, ((i+1) / 2500.0))
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
        IMS.lr.set_value(to_fX(zero_ary + learn_rate))
        IMS.mom_1.set_value(to_fX(zero_ary + momentum))
        IMS.mom_2.set_value(to_fX(zero_ary + 0.99))

        # perform a minibatch update and record the cost for this batch
        Xb = to_fX(Xtr.take(batch_idx, axis=0))
        result = IMS.train_joint(Xb)

        costs = [(costs[j] + result[j]) for j in range(len(result))]
        if ((i % 100) == 0):
            costs = [(v / 100.0) for v in costs]
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
            IMS.save_model_params("IMS_params.pkl")
            # compute a small-sample estimate of NLL bound on validation set
            Xva = row_shuffle(Xva)
            Xb = to_fX(Xva[:5000])
            va_costs = IMS.compute_nll_bound(Xb)
            str1 = "    va_nll_bound : {}".format(va_costs[1])
            str2 = "    va_nll_term  : {}".format(va_costs[2])
            str3 = "    va_kld_q2p   : {}".format(va_costs[3])
            joint_str = "\n".join([str1, str2, str3])
            print(joint_str)
            out_file.write(joint_str+"\n")
            out_file.flush()

if __name__=="__main__":
    test_img_scan()
