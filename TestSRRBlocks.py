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
from SRRBlocks import ImgScan, SeqCondGen
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
        # read_N, write_N = (2, 5) # resolution of reader and writer
        # read_dim = 2*read_N**2   # total number of "pixels" read by reader
        # reader_mlp = AttentionReader(x_dim=x_dim, dec_dim=rnn_dim,
        #                          width=28, height=28,
        #                          N=read_N, **inits)
        att_tag = "YA"
    else:
        read_dim = 2*x_dim
        reader_mlp = Reader(x_dim=x_dim, dec_dim=rnn_dim, **inits)
        att_tag = "NA"
    writer_mlp = MLP([None, None], [rnn_dim, write_dim, x_dim], \
                     name="writer_mlp", **inits)

    # mlps for processing inputs to LSTMs
    con_mlp_in = MLP([Identity()], [                       z_dim, 4*rnn_dim], \
                     name="con_mlp_in", **inits)
    var_mlp_in = MLP([Identity()], [(x_dim + read_dim + rnn_dim), 4*rnn_dim], \
                     name="var_mlp_in", **inits)
    gen_mlp_in = MLP([Identity()], [(x_dim + read_dim + rnn_dim), 4*rnn_dim], \
                     name="gen_mlp_in", **inits)
    mem_mlp_in = MLP([Identity()], [                   2*rnn_dim, 4*rnn_dim], \
                     name="mem_mlp_in", **inits)


    # mlps for turning LSTM outputs into conditionals over z_gen
    gen_mlp_out = CondNet([], [rnn_dim, z_dim], name="gen_mlp_out", **inits)
    var_mlp_out = CondNet([], [rnn_dim, z_dim], name="var_mlp_out", **inits)
    mem_mlp_out = MLP([Identity()], [rnn_dim, 2*rnn_dim], \
                      name="mem_mlp_out", **inits)
    # LSTMs for the actual LSTMs (obviously, perhaps)
    con_rnn = BiasedLSTM(dim=rnn_dim, ig_bias=2.0, fg_bias=2.0, \
                         name="con_rnn", **rnninits)
    gen_rnn = BiasedLSTM(dim=rnn_dim, ig_bias=2.0, fg_bias=2.0, \
                         name="gen_rnn", **rnninits)
    var_rnn = BiasedLSTM(dim=rnn_dim, ig_bias=2.0, fg_bias=2.0, \
                         name="var_rnn", **rnninits)
    mem_rnn = BiasedLSTM(dim=rnn_dim, ig_bias=2.0, fg_bias=2.0, \
                         name="mem_rnn", **rnninits)

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
        mem_mlp_in: preprocesses input to the "memory" LSTM
        mem_rnn: the "memory" LSTM (this stores inter-prediction state)
        mem_mlp_out: emits initial controller state for each prediction
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
                var_rnn=var_rnn,
                mem_mlp_in=mem_mlp_in,
                mem_mlp_out=mem_mlp_out,
                mem_rnn=mem_rnn)
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
    learn_rate = 0.0001
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

#############################################
#############################################
## TEST TIME-LIMITED CONDITIONAL GENERATOR ##
#############################################
#############################################

def test_seq_cond_gen(step_type='add', attention=False):
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
    total_steps = 15
    init_steps = 5
    exit_rate = 0.2
    x_dim = 784
    y_dim = 784
    z_dim = 100
    rnn_dim = 250
    write_dim = 200
    mlp_dim = 250


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
        # read_N, write_N = (2, 5) # resolution of reader and writer
        # read_dim = 2*read_N**2   # total number of "pixels" read by reader
        # reader_mlp = AttentionReader(x_dim=x_dim, dec_dim=rnn_dim,
        #                          width=28, height=28,
        #                          N=read_N, **inits)
        att_tag = "YA"
    else:
        read_dim = 2*x_dim
        reader_mlp = Reader(x_dim=x_dim, dec_dim=rnn_dim, **inits)
        att_tag = "NA"
    writer_mlp = MLP([None, None], [rnn_dim, write_dim, y_dim], \
                     name="writer_mlp", **inits)

    # mlps for processing inputs to LSTMs
    con_mlp_in = MLP([Identity()], [                       z_dim, 4*rnn_dim], \
                     name="con_mlp_in", **inits)
    var_mlp_in = MLP([Identity()], [(y_dim + read_dim + rnn_dim), 4*rnn_dim], \
                     name="var_mlp_in", **inits)
    gen_mlp_in = MLP([Identity()], [        (read_dim + rnn_dim), 4*rnn_dim], \
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

    SeqCondGen_doc_str = \
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

    SCG = SeqCondGen(
                total_steps=total_steps,
                init_steps=init_steps,
                exit_rate=exit_rate,
                step_type=step_type,
                x_dim=x_dim,
                y_dim=y_dim,
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
    SCG.initialize()

    # build the cost gradients, training function, samplers, etc.
    SCG.build_model_funcs()

    #SCG.load_model_params(f_name="SCG_params.pkl")

    ################################################################
    # Apply some updates, to check that they aren't totally broken #
    ################################################################
    print("Beginning to train the model...")
    out_file = open("SCG_results.txt", 'wb')
    out_file.flush()
    costs = [0. for i in range(10)]
    learn_rate = 0.0001
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
        SCG.lr.set_value(to_fX(zero_ary + learn_rate))
        SCG.mom_1.set_value(to_fX(zero_ary + momentum))
        SCG.mom_2.set_value(to_fX(zero_ary + 0.99))

        # perform a minibatch update and record the cost for this batch
        Xb = to_fX(Xtr.take(batch_idx, axis=0))
        result = SCG.train_joint(Xb, Xb)

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
            SCG.save_model_params("SCG_params.pkl")
            # compute a small-sample estimate of NLL bound on validation set
            Xva = row_shuffle(Xva)
            Xb = to_fX(Xva[:1000])
            va_costs = SCG.compute_nll_bound(Xb, Xb)
            str1 = "    va_nll_bound : {}".format(va_costs[1])
            str2 = "    va_nll_term  : {}".format(va_costs[2])
            str3 = "    va_kld_q2p   : {}".format(va_costs[3])
            joint_str = "\n".join([str1, str2, str3])
            print(joint_str)
            out_file.write(joint_str+"\n")
            out_file.flush()

if __name__=="__main__":
    #test_img_scan(attention=False)
    test_seq_cond_gen(step_type='add', attention=False)
