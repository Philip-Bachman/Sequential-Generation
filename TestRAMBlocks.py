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
import time

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
from RAMBlocks import *
from DKCode import get_adam_updates, get_adadelta_updates
from load_data import load_udm, load_tfd, load_svhn_gray, load_binarized_mnist
from HelperFuncs import construct_masked_data, shift_and_scale_into_01, \
                        row_shuffle, to_fX, one_hot_np

RESULT_PATH = "RAM_TEST_RESULTS/"

################################
################################
## TEST COLUMN-WISE GENERATOR ##
################################
################################

def test_oi_seq_cond_gen(attention=False):
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
    outer_steps = 27
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
        read_N = 3 # inner/outer grid dimension for reader
        reader_mlp = SimpleAttentionReader1d(x_dim=x_dim, con_dim=rnn_dim,
                                             N=read_N, init_scale=2.0, **inits)
        read_dim = reader_mlp.read_dim
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
    gen_mlp_in = MLP([Identity()], [        (read_dim + rnn_dim), 4*rnn_dim], \
                     name="gen_mlp_in", **inits)
    var_mlp_in = MLP([Identity()], [(x_dim + read_dim + rnn_dim), 4*rnn_dim], \
                     name="var_mlp_in", **inits)
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

    OISeqCondGen_doc_str = \
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

    IMS = OISeqCondGen(
                obs_dim=x_dim,
                outer_steps=outer_steps,
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
                mem_rnn=mem_rnn
    )
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
        Xb = Xb.reshape(batch_size, x_dim, x_dim).swapaxes(0,2).swapaxes(1,2)
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
            Xb = Xb.reshape(batch_size, x_dim, x_dim).swapaxes(0,2).swapaxes(1,2)
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

def test_seq_cond_gen_static(step_type='add'):
    ##############################
    # File tag, for output stuff #
    ##############################
    result_tag = "{}AAA_SCG".format(RESULT_PATH)

    ##########################
    # Get some training data #
    ##########################
    rng = np.random.RandomState(1234)
    dataset = 'data/mnist.pkl.gz'
    datasets = load_udm(dataset, as_shared=False, zero_mean=False)
    # get training/validation/test images
    Xtr = datasets[0][0]
    Xva = datasets[1][0]
    Xte = datasets[2][0]
    Xtr = to_fX(shift_and_scale_into_01(Xtr))
    Xva = to_fX(shift_and_scale_into_01(Xva))
    Xte = to_fX(shift_and_scale_into_01(Xte))
    obs_dim = Xtr.shape[1]
    # get label representations
    y_reps = 10
    Ytr = one_hot_np(datasets[0][1]-1, cat_dim=10).repeat(y_reps, axis=1)
    Yva = one_hot_np(datasets[1][1]-1, cat_dim=10).repeat(y_reps, axis=1)
    Yte = one_hot_np(datasets[2][1]-1, cat_dim=10).repeat(y_reps, axis=1)
    label_dim = Ytr.shape[1]
    # merge image and lagel representations
    print("Xtr.shape: {}".format(Xtr.shape))
    print("Ytr.shape: {}".format(Ytr.shape))
    XYtr = to_fX( np.hstack( [Xtr, Ytr] ) )
    XYva = to_fX( np.hstack( [Xva, Yva] ) )
    tr_samples = XYtr.shape[0]
    va_samples = XYva.shape[0]
    batch_size = 200

    def split_xy(xy_ary):
        x_ary = xy_ary[:,:obs_dim]
        y_ary = xy_ary[:,obs_dim:]
        return x_ary, y_ary

    ############################################################
    # Setup some parameters for the Iterative Refinement Model #
    ############################################################
    total_steps = 10
    init_steps = 3
    exit_rate = 0.2
    x_dim = obs_dim
    y_dim = obs_dim + label_dim
    z_dim = 100
    rnn_dim = 400
    write_dim = 400
    mlp_dim = 400

    def visualize_attention(result, pre_tag="AAA", post_tag="AAA"):
        seq_len = result[0].shape[0]
        samp_count = result[0].shape[1]
        # get generated predictions
        x_samps = np.zeros((seq_len*samp_count, obs_dim))
        y_samps = np.zeros((seq_len*samp_count, label_dim))
        idx = 0
        for s1 in range(samp_count):
            for s2 in range(seq_len):
                x_samps[idx] = result[0][s2,s1,:obs_dim]
                y_samps[idx] = result[0][s2,s1,obs_dim:]
                # add ticks at the corners of label predictions, to make them
                # easier to parse visually.
                max_val = np.mean(result[0][s2,s1,obs_dim:])
                y_samps[idx][0] = max_val
                y_samps[idx][9] = max_val
                y_samps[idx][-1] = max_val
                y_samps[idx][-10] = max_val
                idx += 1
        file_name = "{0:s}_traj_xs_{1:s}.png".format(pre_tag, post_tag)
        utils.visualize_samples(x_samps, file_name, num_rows=20)
        file_name = "{0:s}_traj_ys_{1:s}.png".format(pre_tag, post_tag)
        utils.visualize_samples(y_samps, file_name, num_rows=20)
        # get sequential attention maps
        seq_samps = np.zeros((seq_len*samp_count, x_dim))
        idx = 0
        for s1 in range(samp_count):
            for s2 in range(seq_len):
                seq_samps[idx] = result[1][s2,s1,:x_dim] + result[1][s2,s1,x_dim:]
                idx += 1
        file_name = "{0:s}_traj_att_maps_{1:s}.png".format(pre_tag, post_tag)
        utils.visualize_samples(seq_samps, file_name, num_rows=20)
        # get sequential attention maps (read out values)
        seq_samps = np.zeros((seq_len*samp_count, x_dim))
        idx = 0
        for s1 in range(samp_count):
            for s2 in range(seq_len):
                seq_samps[idx] = result[2][s2,s1,:x_dim] + result[2][s2,s1,x_dim:]
                idx += 1
        file_name = "{0:s}_traj_read_outs_{1:s}.png".format(pre_tag, post_tag)
        utils.visualize_samples(seq_samps, file_name, num_rows=20)
        return

    rnninits = {
        'weights_init': IsotropicGaussian(0.01),
        'biases_init': Constant(0.),
    }
    inits = {
        'weights_init': IsotropicGaussian(0.01),
        'biases_init': Constant(0.),
    }

    read_N = 2 # inner/outer grid dimension for reader
    reader_mlp = SimpleAttentionReader2d(x_dim=x_dim, con_dim=rnn_dim,
                                         width=28, height=28, N=read_N,
                                         init_scale=2.0, **inits)
    read_dim = reader_mlp.read_dim # total number of "pixels" read by reader

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
    con_mlp_out = CondNet([Rectifier(), Rectifier()], \
                          [rnn_dim, mlp_dim, mlp_dim, z_dim], \
                          name="con_mlp_out", **inits)
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

    SCG = SeqCondGen(
                x_and_y_are_seqs=False, # this test doesn't use sequential x/y
                total_steps=total_steps,
                init_steps=init_steps,
                exit_rate=exit_rate,
                nll_weight=0.0, # ignored, because x_and_y_are_seqs == False
                step_type=step_type,
                x_dim=x_dim,
                y_dim=y_dim,
                reader_mlp=reader_mlp,
                writer_mlp=writer_mlp,
                con_mlp_in=con_mlp_in,
                con_mlp_out=con_mlp_out,
                con_rnn=con_rnn,
                gen_mlp_in=gen_mlp_in,
                gen_mlp_out=gen_mlp_out,
                gen_rnn=gen_rnn,
                var_mlp_in=var_mlp_in,
                var_mlp_out=var_mlp_out,
                var_rnn=var_rnn)
    SCG.initialize()

    compile_start_time = time.time()

    # build the attention trajectory sampler
    SCG.build_attention_funcs()

    # quick test of attention trajectory sampler
    samp_count = 100
    XYb = XYva[:samp_count,:]
    Xb, Yb = split_xy(XYb)
    #Xb = Xva[:samp_count]
    result = SCG.sample_attention(Xb, XYb)
    visualize_attention(result, pre_tag=result_tag, post_tag="b0")

    # build the main model functions (i.e. training and cost functions)
    SCG.build_model_funcs()

    compile_end_time = time.time()
    compile_minutes = (compile_end_time - compile_start_time) / 60.0
    print("THEANO COMPILE TIME (MIN): {}".format(compile_minutes))

    #SCG.load_model_params(f_name="SCG_params.pkl")

    ################################################################
    # Apply some updates, to check that they aren't totally broken #
    ################################################################
    print("Beginning to train the model...")
    out_file = open("{}_results.txt".format(result_tag), 'wb')
    out_file.flush()
    costs = [0. for i in range(10)]
    learn_rate = 0.0001
    momentum = 0.8
    batch_idx = np.arange(batch_size) + tr_samples
    for i in range(250000):
        scale = min(1.0, ((i+1) / 2500.0))
        if (((i + 1) % 10000) == 0):
            learn_rate = learn_rate * 0.95
        if (i > 10000):
            momentum = 0.95
        else:
            momentum = 0.8
        # get the indices of training samples for this batch update
        batch_idx += batch_size
        if (np.max(batch_idx) >= tr_samples):
            # we finished an "epoch", so we rejumble the training set
            XYtr = row_shuffle(XYtr)
            #Xtr = row_shuffle(Xtr)
            batch_idx = np.arange(batch_size)
        # set sgd and objective function hyperparams for this update
        SCG.set_sgd_params(lr=learn_rate, mom_1=momentum, mom_2=0.99)
        SCG.set_lam_kld(lam_kld_q2p=0.95, lam_kld_p2q=0.05, lam_kld_p2g=0.1)
        # perform a minibatch update and record the cost for this batch
        XYb = XYtr.take(batch_idx, axis=0)
        Xb, Yb = split_xy(XYb)
        #Xb = Xtr.take(batch_idx, axis=0)
        result = SCG.train_joint(Xb, XYb)
        costs = [(costs[j] + result[j]) for j in range(len(result))]

        # output diagnostic information and checkpoint parameters, etc.
        if ((i % 250) == 0):
            costs = [(v / 250.0) for v in costs]
            str1 = "-- batch {0:d} --".format(i)
            str2 = "    total_cost: {0:.4f}".format(costs[0])
            str3 = "    nll_bound : {0:.4f}".format(costs[1])
            str4 = "    nll_term  : {0:.4f}".format(costs[2])
            str5 = "    kld_q2p   : {0:.4f}".format(costs[3])
            str6 = "    kld_p2q   : {0:.4f}".format(costs[4])
            str7 = "    kld_p2g   : {0:.4f}".format(costs[5])
            str8 = "    reg_term  : {0:.4f}".format(costs[6])
            joint_str = "\n".join([str1, str2, str3, str4, str5, str6, str7, str8])
            print(joint_str)
            out_file.write(joint_str+"\n")
            out_file.flush()
            costs = [0.0 for v in costs]
        if ((i % 500) == 0): #((i % 1000) == 0):
            SCG.save_model_params("{}_params.pkl".format(result_tag))
            # compute a small-sample estimate of NLL bound on validation set
            XYva = row_shuffle(XYva)
            XYb = XYva[:1000]
            Xb, Yb = split_xy(XYb)
            #Xva = row_shuffle(Xva)
            #Xb = Xva[:1000]
            va_costs = SCG.compute_nll_bound(Xb, XYb)
            str1 = "    va_nll_bound : {}".format(va_costs[1])
            str2 = "    va_nll_term  : {}".format(va_costs[2])
            str3 = "    va_kld_q2p   : {}".format(va_costs[3])
            joint_str = "\n".join([str1, str2, str3])
            print(joint_str)
            out_file.write(joint_str+"\n")
            out_file.flush()
            ###########################################
            # Sample and draw attention trajectories. #
            ###########################################
            samp_count = 100
            XYb = XYva[:samp_count,:]
            Xb, Yb = split_xy(XYb)
            #Xb = Xva[:samp_count]
            result = SCG.sample_attention(Xb, XYb)
            post_tag = "b{0:d}".format(i)
            visualize_attention(result, pre_tag=result_tag, post_tag=post_tag)

##########################################################
##########################################################
## TEST SEQUENCE-STYLE PREDICTION AND ACTIVE PERCEPTION ##
##########################################################
##########################################################

def test_seq_cond_gen_sequence(step_type='add'):
    ##############################
    # File tag, for output stuff #
    ##############################
    result_tag = "{}BBB_SCG".format(RESULT_PATH)

    ##########################
    # Get some training data #
    ##########################
    rng = np.random.RandomState(1234)
    dataset = 'data/mnist.pkl.gz'
    datasets = load_udm(dataset, as_shared=False, zero_mean=False)
    # get training/validation/test images
    Xtr = datasets[0][0]
    Xva = datasets[1][0]
    Xte = datasets[2][0]
    Xtr = to_fX(shift_and_scale_into_01(Xtr))
    Xva = to_fX(shift_and_scale_into_01(Xva))
    Xte = to_fX(shift_and_scale_into_01(Xte))
    obs_dim = Xtr.shape[1]
    # get label representations
    tr_samples = Xtr.shape[0]
    va_samples = Xva.shape[0]
    batch_size = 200
    step_reps = 3

    ############################################################
    # Setup some parameters for the Iterative Refinement Model #
    ############################################################
    total_steps = step_reps * 28
    init_steps = step_reps
    exit_rate = 0.0
    nll_weight = 1.0 / step_reps
    x_dim = 28
    y_dim = 28
    z_dim = 100
    rnn_dim = 300
    write_dim = 250
    mlp_dim = 250

    def visualize_attention(sampler_result, pre_tag="AAA", post_tag="AAA"):
        # get generated predictions
        seq_len = sampler_result[0].shape[0]
        samp_count = sampler_result[0].shape[1]
        x_dim = sampler_result[0].shape[2]
        seq_samps = np.zeros((samp_count, 28*28))
        for samp in range(samp_count):
            step = 0
            samp_vals = np.zeros((28,28))
            for col in range(28):
                col_vals = np.zeros((28,))
                for rep in range(step_reps):
                    if (rep == (step_reps-1)):
                        col_vals = sampler_result[0][step,samp,:]
                    step += 1
                samp_vals[:,col] = col_vals
            seq_samps[samp,:] = samp_vals.ravel()
        file_name = "{0:s}_traj_xs_{1:s}.png".format(pre_tag, post_tag)
        utils.visualize_samples(seq_samps, file_name, num_rows=10)
        # get sequential attention maps
        seq_samps = np.zeros((samp_count, 28*28))
        for samp in range(samp_count):
            step = 0
            samp_vals = np.zeros((28,28))
            for col in range(28):
                col_vals = np.zeros((28,))
                for rep in range(step_reps):
                    col_vals = col_vals + sampler_result[1][step,samp,:x_dim]
                    col_vals = col_vals + sampler_result[1][step,samp,x_dim:]
                    step += 1
                samp_vals[:,col] = col_vals / (2.0*step_reps)
            seq_samps[samp,:] = samp_vals.ravel()
        file_name = "{0:s}_traj_att_maps_{1:s}.png".format(pre_tag, post_tag)
        utils.visualize_samples(seq_samps, file_name, num_rows=10)
        # get sequential attention maps (read out values)
        seq_samps = np.zeros((samp_count, 28*28))
        for samp in range(samp_count):
            step = 0
            samp_vals = np.zeros((28,28))
            for col in range(28):
                col_vals = np.zeros((28,))
                for rep in range(step_reps):
                    col_vals = col_vals + sampler_result[2][step,samp,:x_dim]
                    col_vals = col_vals + sampler_result[2][step,samp,x_dim:]
                    step += 1
                samp_vals[:,col] = col_vals / (2.0*step_reps)
            seq_samps[samp,:] = samp_vals.ravel()
        file_name = "{0:s}_traj_read_outs_{1:s}.png".format(pre_tag, post_tag)
        utils.visualize_samples(seq_samps, file_name, num_rows=10)
        return

    def batch_reshape(Xb, reps=step_reps):
        # reshape for stuff
        bs = Xb.shape[0]
        xb = Xb.reshape((bs, 28, 28)).swapaxes(0,2).swapaxes(1,2)
        xb = xb.repeat(reps, axis=0)
        return xb

    rnninits = {
        'weights_init': IsotropicGaussian(0.01),
        'biases_init': Constant(0.),
    }
    inits = {
        'weights_init': IsotropicGaussian(0.01),
        'biases_init': Constant(0.),
    }

    read_N = 2 # inner/outer grid dimension for reader
    read_dim = 2*read_N   # total number of "pixels" read by reader
    reader_mlp = SimpleAttentionReader1d(x_dim=x_dim, con_dim=rnn_dim,
                                         N=read_N, init_scale=2.0, **inits)

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
    con_mlp_out = CondNet([], [rnn_dim, z_dim], name="con_mlp_out", **inits)
    gen_mlp_out = CondNet([], [rnn_dim, z_dim], name="gen_mlp_out", **inits)
    var_mlp_out = CondNet([], [rnn_dim, z_dim], name="var_mlp_out", **inits)

    # LSTMs for the actual LSTMs (obviously, perhaps)
    con_rnn = BiasedLSTM(dim=rnn_dim, ig_bias=2.0, fg_bias=2.0, \
                         name="con_rnn", **rnninits)
    gen_rnn = BiasedLSTM(dim=rnn_dim, ig_bias=2.0, fg_bias=2.0, \
                         name="gen_rnn", **rnninits)
    var_rnn = BiasedLSTM(dim=rnn_dim, ig_bias=2.0, fg_bias=2.0, \
                         name="var_rnn", **rnninits)

    SCG = SeqCondGen(
                x_and_y_are_seqs=True, # this test uses sequential x/y
                total_steps=total_steps,
                init_steps=init_steps,
                exit_rate=exit_rate,
                nll_weight=nll_weight,
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

    compile_start_time = time.time()

    # build the cost gradients, training function, samplers, etc.
    SCG.build_attention_funcs()

    ###########################################
    # Sample and draw attention trajectories. #
    ###########################################
    samp_count = 100
    Xb = Xva[:samp_count,:]
    Xb = batch_reshape(Xb, reps=step_reps)
    print("Xb.shape: {}".format(Xb.shape))
    result = SCG.sample_attention(Xb, Xb)
    visualize_attention(result, pre_tag=result_tag, post_tag="b0")
    print("TESTED SAMPLER!")
    Xva = row_shuffle(Xva)
    Xb = Xva[:500]
    Xb = batch_reshape(Xb, reps=step_reps)
    va_costs = SCG.simple_nll_bound(Xb, Xb)
    print("nll_bound : {}".format(va_costs[0]))
    print("nll_term  : {}".format(va_costs[1]))
    print("kld_q2p   : {}".format(va_costs[2]))
    print("TESTED NLL BOUND!")

    SCG.build_model_funcs()

    compile_end_time = time.time()
    compile_minutes = (compile_end_time - compile_start_time) / 60.0
    print("THEANO COMPILE TIME (MIN): {}".format(compile_minutes))

    #SCG.load_model_params(f_name="SCG_params.pkl")

    ################################################################
    # Apply some updates, to check that they aren't totally broken #
    ################################################################
    print("Beginning to train the model...")
    out_file = open("{}_results.txt".format(result_tag), 'wb')
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
        SCG.set_sgd_params(lr=learn_rate, mom_1=momentum, mom_2=0.99)
        SCG.set_lam_kld(lam_kld_q2p=0.95, lam_kld_p2q=0.05, lam_kld_p2g=0.0)
        # perform a minibatch update and record the cost for this batch
        Xb = Xtr.take(batch_idx, axis=0)
        Xb = batch_reshape(Xb, reps=step_reps)
        result = SCG.train_joint(Xb, Xb)
        costs = [(costs[j] + result[j]) for j in range(len(result))]

        # output diagnostic information and checkpoint parameters, etc.
        if ((i % 250) == 0):
            costs = [(v / 250.0) for v in costs]
            str1 = "-- batch {0:d} --".format(i)
            str2 = "    total_cost: {0:.4f}".format(costs[0])
            str3 = "    nll_bound : {0:.4f}".format(costs[1])
            str4 = "    nll_term  : {0:.4f}".format(costs[2])
            str5 = "    kld_q2p   : {0:.4f}".format(costs[3])
            str6 = "    kld_p2q   : {0:.4f}".format(costs[4])
            str7 = "    kld_p2g   : {0:.4f}".format(costs[5])
            str8 = "    reg_term  : {0:.4f}".format(costs[6])
            joint_str = "\n".join([str1, str2, str3, str4, str5, str6, str7, str8])
            print(joint_str)
            out_file.write(joint_str+"\n")
            out_file.flush()
            costs = [0.0 for v in costs]
        if ((i % 500) == 0): #((i % 1000) == 0):
            SCG.save_model_params("{}_params.pkl".format(result_tag))
            # compute a small-sample estimate of NLL bound on validation set
            Xva = row_shuffle(Xva)
            Xb = Xva[:500]
            Xb = batch_reshape(Xb, reps=step_reps)
            va_costs = SCG.compute_nll_bound(Xb, Xb)
            str1 = "    va_nll_bound : {}".format(va_costs[1])
            str2 = "    va_nll_term  : {}".format(va_costs[2])
            str3 = "    va_kld_q2p   : {}".format(va_costs[3])
            joint_str = "\n".join([str1, str2, str3])
            print(joint_str)
            out_file.write(joint_str+"\n")
            out_file.flush()
            ###########################################
            # Sample and draw attention trajectories. #
            ###########################################
            post_tag = "b{}".format(i)
            Xb = Xva[:100,:]
            Xb = batch_reshape(Xb, reps=step_reps)
            result = SCG.sample_attention(Xb, Xb)
            visualize_attention(result, pre_tag=result_tag, post_tag=post_tag)


if __name__=="__main__":
    #test_oi_seq_cond_gen(attention=False)
    test_seq_cond_gen_static(step_type='add')
    #test_seq_cond_gen_sequence(step_type='add')
