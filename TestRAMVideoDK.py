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
from blocks.bricks import Tanh, Identity, Rectifier, MLP
from blocks.bricks.cost import BinaryCrossEntropy
from blocks.bricks.recurrent import SimpleRecurrent, LSTM

# phil's sweetness
import utils
from BlocksModels import *
from RAMBlocks import *
from SeqCondGenVariants import *
from DKCode import get_adam_updates, get_adadelta_updates
from load_data import load_udm, load_tfd, load_svhn_gray, load_binarized_mnist
from HelperFuncs import sample_data_masks, shift_and_scale_into_01, \
                        row_shuffle, to_fX, one_hot_np
from MotionRenderers import TrajectoryGenerator, ObjectPainter

RESULT_PATH = "RAM_TEST_RESULTS/"


###########################################
###########################################
##                                       ##
## Test attention-based image "copying". ##
##                                       ##
###########################################
###########################################

def test_seq_cond_gen_copy(step_type='add', res_tag="AAA"):
    ##############################
    # File tag, for output stuff #
    ##############################
    result_tag = "{}TEST_{}".format(RESULT_PATH, res_tag)

    ##########################
    # Get some training data #
    ##########################
    rng = np.random.RandomState(1234)
    dataset = 'data/mnist.pkl.gz'
    datasets = load_udm(dataset, as_shared=False, zero_mean=False)
    Xtr = datasets[0][0]
    Xva = datasets[1][0]
    Xte = datasets[2][0]
    # merge validation set and training set, and test on test set.
    #Xtr = np.concatenate((Xtr, Xva), axis=0)
    #Xva = Xte
    Xtr = to_fX(shift_and_scale_into_01(Xtr))
    Xva = to_fX(shift_and_scale_into_01(Xva))

    # basic params
    batch_size = 128
    traj_len = 20
    im_dim = 28
    obs_dim = im_dim*im_dim

    def sample_batch(np_ary, bs=100):
        row_count = np_ary.shape[0]
        samp_idx = npr.randint(low=0,high=row_count,size=(bs,))
        xb = np_ary.take(samp_idx, axis=0)
        return xb

    ############################################################
    # Setup some parameters for the Iterative Refinement Model #
    ############################################################
    total_steps = traj_len
    init_steps = 5
    exit_rate = 0.1
    nll_weight = 0.0
    x_dim = obs_dim
    y_dim = obs_dim
    z_dim = 128
    att_spec_dim = 5
    rnn_dim = 512
    mlp_dim = 512

    def visualize_attention(result, pre_tag="AAA", post_tag="AAA"):
        seq_len = result[0].shape[0]
        samp_count = result[0].shape[1]
        # get generated predictions
        x_samps = np.zeros((seq_len*samp_count, obs_dim))
        idx = 0
        for s1 in range(samp_count):
            for s2 in range(seq_len):
                x_samps[idx] = result[0][s2,s1,:]
                idx += 1
        file_name = "{0:s}_traj_xs_{1:s}.png".format(pre_tag, post_tag)
        utils.visualize_samples(x_samps, file_name, num_rows=samp_count)
        # get sequential attention maps
        seq_samps = np.zeros((seq_len*samp_count, obs_dim))
        idx = 0
        for s1 in range(samp_count):
            for s2 in range(seq_len):
                seq_samps[idx] = result[1][s2,s1,:]
                idx += 1
        file_name = "{0:s}_traj_att_maps_{1:s}.png".format(pre_tag, post_tag)
        utils.visualize_samples(seq_samps, file_name, num_rows=samp_count)
        # get sequential attention maps (read out values)
        seq_samps = np.zeros((seq_len*samp_count, obs_dim))
        idx = 0
        for s1 in range(samp_count):
            for s2 in range(seq_len):
                seq_samps[idx] = result[2][s2,s1,:]
                idx += 1
        file_name = "{0:s}_traj_read_outs_{1:s}.png".format(pre_tag, post_tag)
        utils.visualize_samples(seq_samps, file_name, num_rows=samp_count)
        # get original input sequences
        seq_samps = np.zeros((seq_len*samp_count, obs_dim))
        idx = 0
        for s1 in range(samp_count):
            for s2 in range(seq_len):
                seq_samps[idx] = result[3][s2,s1,:]
                idx += 1
        file_name = "{0:s}_traj_xs_in_{1:s}.png".format(pre_tag, post_tag)
        utils.visualize_samples(seq_samps, file_name, num_rows=samp_count)
        return

    rnninits = {
        'weights_init': IsotropicGaussian(0.01),
        'biases_init': Constant(0.),
    }
    inits = {
        'weights_init': IsotropicGaussian(0.01),
        'biases_init': Constant(0.),
    }

    # module for doing local 2d read defined by an attention specification
    img_scale = 1.0 # image coords will range over [-img_scale...img_scale]
    read_N = 2      # use NxN grid for reader
    reader_mlp = FovAttentionReader2d(x_dim=obs_dim,
                                      width=im_dim, height=im_dim, N=read_N,
                                      img_scale=img_scale, att_scale=0.5,
                                      **inits)
    read_dim = reader_mlp.read_dim # total number of "pixels" read by reader

    # MLP for updating belief state based on con_rnn
    writer_mlp = MLP([None, None], [rnn_dim, mlp_dim, obs_dim], \
                     name="writer_mlp", **inits)

    # mlps for processing inputs to LSTMs
    con_mlp_in = MLP([Identity()], \
                     [                       z_dim, 4*rnn_dim], \
                     name="con_mlp_in", **inits)
    var_mlp_in = MLP([Identity()], \
                     [(read_dim + read_dim + att_spec_dim + rnn_dim), 4*rnn_dim], \
                     name="var_mlp_in", **inits)
    gen_mlp_in = MLP([Identity()], \
                     [        (read_dim + att_spec_dim + rnn_dim), 4*rnn_dim], \
                     name="gen_mlp_in", **inits)

    # mlps for turning LSTM outputs into conditionals over z_gen
    con_mlp_out = CondNet([], [rnn_dim, att_spec_dim], \
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

    SCG = SeqCondGenRAM(
                x_and_y_are_seqs=False,
                total_steps=total_steps,
                init_steps=init_steps,
                exit_rate=exit_rate,
                nll_weight=nll_weight,
                step_type=step_type,
                x_dim=obs_dim,
                y_dim=obs_dim,
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
    Xb = sample_batch(Xtr, bs=32)
    result = SCG.sample_attention(Xb, Xb)
    visualize_attention(result, pre_tag=result_tag, post_tag="b0")

    # build the main model functions (i.e. training and cost functions)
    SCG.build_model_funcs()

    compile_end_time = time.time()
    compile_minutes = (compile_end_time - compile_start_time) / 60.0
    print("THEANO COMPILE TIME (MIN): {}".format(compile_minutes))

    # TEST SAVE/LOAD FUNCTIONALITY
    param_save_file = "{}_params.pkl".format(result_tag)
    SCG.save_model_params(param_save_file)
    SCG.load_model_params(param_save_file)

    ################################################################
    # Apply some updates, to check that they aren't totally broken #
    ################################################################
    print("Beginning to train the model...")
    out_file = open("{}_results.txt".format(result_tag), 'wb')
    out_file.flush()
    costs = [0. for i in range(10)]
    learn_rate = 0.0001
    momentum = 0.95
    for i in range(250000):
        lr_scale = min(1.0, ((i+1) / 5000.0))
        mom_scale = min(1.0, ((i+1) / 10000.0))
        if (((i + 1) % 10000) == 0):
            learn_rate = learn_rate * 0.95
        # set sgd and objective function hyperparams for this update
        SCG.set_sgd_params(lr=lr_scale*learn_rate, mom_1=mom_scale*momentum, mom_2=0.99)
        SCG.set_lam_kld(lam_kld_q2p=0.95, lam_kld_p2q=0.05, \
                        lam_kld_amu=0.0, lam_kld_alv=0.1)
        # perform a minibatch update and record the cost for this batch
        Xb = sample_batch(Xtr, bs=batch_size)
        result = SCG.train_joint(Xb, Xb)
        costs = [(costs[j] + result[j]) for j in range(len(result))]
        # output diagnostic information and checkpoint parameters, etc.
        if ((i % 250) == 0):
            costs = [(v / 250.0) for v in costs]
            str1 = "-- batch {0:d} --".format(i)
            str2 = "    total_cost: {0:.4f}".format(costs[0])
            str3 = "    nll_term  : {0:.4f}".format(costs[1])
            str4 = "    kld_q2p   : {0:.4f}".format(costs[2])
            str5 = "    kld_p2q   : {0:.4f}".format(costs[3])
            str6 = "    kld_amu   : {0:.4f}".format(costs[4])
            str7 = "    kld_alv   : {0:.4f}".format(costs[5])
            str8 = "    reg_term  : {0:.4f}".format(costs[6])
            joint_str = "\n".join([str1, str2, str3, str4, str5, str6, str7, str8])
            print(joint_str)
            out_file.write(joint_str+"\n")
            out_file.flush()
            costs = [0.0 for v in costs]
        if ((i % 500) == 0):
            SCG.save_model_params("{}_params.pkl".format(result_tag))
            #############################################
            # check model performance on validation set #
            #############################################
            Xb = sample_batch(Xva, bs=500)
            result = SCG.compute_nll_bound(Xb, Xb)
            str2 = "    va_total_cost: {0:.4f}".format(float(result[0]))
            str3 = "    va_nll_term  : {0:.4f}".format(float(result[1]))
            str4 = "    va_kld_q2p   : {0:.4f}".format(float(result[2]))
            str5 = "    va_kld_p2q   : {0:.4f}".format(float(result[3]))
            str6 = "    va_kld_amu   : {0:.4f}".format(float(result[4]))
            str7 = "    va_kld_alv   : {0:.4f}".format(float(result[5]))
            str8 = "    va_reg_term  : {0:.4f}".format(float(result[6]))
            joint_str = "\n".join([str2, str3, str4, str5, str6, str7, str8])
            print(joint_str)
            out_file.write(joint_str+"\n")
            out_file.flush()
            ###########################################
            # sample and draw attention trajectories. #
            ###########################################
            Xb = sample_batch(Xva, bs=32)
            result = SCG.sample_attention(Xb, Xb)
            post_tag = "b{0:d}".format(i)
            visualize_attention(result, pre_tag=result_tag, post_tag=post_tag)


######################################
######################################
##                                  ##
## Test attention-based imputation. ##
##                                  ##
######################################
######################################

def test_seq_cond_gen_impute(step_type='add', res_tag="AAA"):
    ##############################
    # File tag, for output stuff #
    ##############################
    result_tag = "{}TEST_{}".format(RESULT_PATH, res_tag)

    ##########################
    # Get some training data #
    ##########################
    rng = np.random.RandomState(1234)
    dataset = 'data/mnist.pkl.gz'
    datasets = load_udm(dataset, as_shared=False, zero_mean=False)
    Xtr = datasets[0][0]
    Xva = datasets[1][0]
    Xte = datasets[2][0]
    # merge validation set and training set, and test on test set.
    #Xtr = np.concatenate((Xtr, Xva), axis=0)
    #Xva = Xte
    Xtr = to_fX(shift_and_scale_into_01(Xtr))
    Xva = to_fX(shift_and_scale_into_01(Xva))

    # basic params
    drop_prob = 0.0
    occ_dim = 16
    batch_size = 128
    traj_len = 15
    im_dim = 28
    obs_dim = im_dim*im_dim

    def sample_batch(np_ary, bs=100):
        row_count = np_ary.shape[0]
        samp_idx = npr.randint(low=0,high=row_count,size=(bs,))
        xb = np_ary.take(samp_idx, axis=0)
        return xb

    ############################################################
    # Setup some parameters for the Iterative Refinement Model #
    ############################################################
    total_steps = traj_len
    init_steps = 3
    exit_rate = 0.2
    nll_weight = 0.0
    x_dim = obs_dim
    y_dim = obs_dim
    z_dim = 128
    att_spec_dim = 5
    rnn_dim = 512
    mlp_dim = 512

    def visualize_attention(result, pre_tag="AAA", post_tag="AAA"):
        seq_len = result[0].shape[0]
        samp_count = result[0].shape[1]
        # get generated predictions
        x_samps = np.zeros((seq_len*samp_count, obs_dim))
        idx = 0
        for s1 in range(samp_count):
            for s2 in range(seq_len):
                x_samps[idx] = result[0][s2,s1,:]
                idx += 1
        file_name = "{0:s}_traj_xs_{1:s}.png".format(pre_tag, post_tag)
        utils.visualize_samples(x_samps, file_name, num_rows=samp_count)
        # get sequential attention maps
        seq_samps = np.zeros((seq_len*samp_count, obs_dim))
        idx = 0
        for s1 in range(samp_count):
            for s2 in range(seq_len):
                seq_samps[idx] = result[1][s2,s1,:]
                idx += 1
        file_name = "{0:s}_traj_att_maps_{1:s}.png".format(pre_tag, post_tag)
        utils.visualize_samples(seq_samps, file_name, num_rows=samp_count)
        # get sequential attention maps (read out values)
        seq_samps = np.zeros((seq_len*samp_count, obs_dim))
        idx = 0
        for s1 in range(samp_count):
            for s2 in range(seq_len):
                seq_samps[idx] = result[2][s2,s1,:]
                idx += 1
        file_name = "{0:s}_traj_read_outs_{1:s}.png".format(pre_tag, post_tag)
        utils.visualize_samples(seq_samps, file_name, num_rows=samp_count)
        # get original input sequences
        seq_samps = np.zeros((seq_len*samp_count, obs_dim))
        idx = 0
        for s1 in range(samp_count):
            for s2 in range(seq_len):
                seq_samps[idx] = result[3][s2,s1,:]
                idx += 1
        file_name = "{0:s}_traj_xs_in_{1:s}.png".format(pre_tag, post_tag)
        utils.visualize_samples(seq_samps, file_name, num_rows=samp_count)
        return

    rnninits = {
        'weights_init': IsotropicGaussian(0.01),
        'biases_init': Constant(0.),
    }
    inits = {
        'weights_init': IsotropicGaussian(0.01),
        'biases_init': Constant(0.),
    }

    # module for doing local 2d read defined by an attention specification
    img_scale = 1.0 # image coords will range over [-img_scale...img_scale]
    read_N = 2      # use NxN grid for reader
    reader_mlp = FovAttentionReader2d(x_dim=obs_dim,
                                      width=im_dim, height=im_dim, N=read_N,
                                      img_scale=img_scale, att_scale=0.5,
                                      **inits)
    read_dim = reader_mlp.read_dim # total number of "pixels" read by reader

    # MLP for updating belief state based on con_rnn
    writer_mlp = MLP([None, None], [rnn_dim, mlp_dim, obs_dim], \
                     name="writer_mlp", **inits)

    # mlps for processing inputs to LSTMs
    con_mlp_in = MLP([Identity()], \
                     [                       z_dim, 4*rnn_dim], \
                     name="con_mlp_in", **inits)
    var_mlp_in = MLP([Identity()], \
                     [(read_dim + read_dim + att_spec_dim + rnn_dim), 4*rnn_dim], \
                     name="var_mlp_in", **inits)
    gen_mlp_in = MLP([Identity()], \
                     [        (read_dim + att_spec_dim + rnn_dim), 4*rnn_dim], \
                     name="gen_mlp_in", **inits)

    # mlps for turning LSTM outputs into conditionals over z_gen
    con_mlp_out = CondNet([], [rnn_dim, att_spec_dim], \
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

    SCG = SeqCondGenIMP(
                x_and_y_are_seqs=False,
                total_steps=total_steps,
                init_steps=init_steps,
                exit_rate=exit_rate,
                nll_weight=nll_weight,
                step_type=step_type,
                x_dim=obs_dim,
                y_dim=obs_dim,
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
                var_rnn=var_rnn,
                att_noise=0.1)
    SCG.initialize()

    compile_start_time = time.time()

    # build the attention trajectory sampler
    SCG.build_attention_funcs()

    # quick test of attention trajectory sampler
    Xb = sample_batch(Xtr, bs=32)
    Mb = sample_data_masks(Xb, drop_prob=drop_prob, occ_dim=occ_dim)
    result = SCG.sample_attention(Xb, Mb)
    visualize_attention(result, pre_tag=result_tag, post_tag="b0")

    # build the main model functions (i.e. training and cost functions)
    SCG.build_model_funcs()

    compile_end_time = time.time()
    compile_minutes = (compile_end_time - compile_start_time) / 60.0
    print("THEANO COMPILE TIME (MIN): {}".format(compile_minutes))

    # TEST SAVE/LOAD FUNCTIONALITY
    param_save_file = "{}_params.pkl".format(result_tag)
    SCG.save_model_params(param_save_file)
    SCG.load_model_params(param_save_file)

    ################################################################
    # Apply some updates, to check that they aren't totally broken #
    ################################################################
    print("Beginning to train the model...")
    out_file = open("{}_results.txt".format(result_tag), 'wb')
    out_file.flush()
    costs = [0. for i in range(10)]
    learn_rate = 0.0001
    momentum = 0.95
    for i in range(250000):
        lr_scale = min(1.0, ((i+1) / 5000.0))
        mom_scale = min(1.0, ((i+1) / 10000.0))
        if (((i + 1) % 10000) == 0):
            learn_rate = learn_rate * 0.95
        # set sgd and objective function hyperparams for this update
        SCG.set_sgd_params(lr=lr_scale*learn_rate, mom_1=mom_scale*momentum, mom_2=0.99)
        SCG.set_lam_kld(lam_kld_q2p=0.95, lam_kld_p2q=0.05, \
                        lam_kld_amu=0.0, lam_kld_alv=0.1)
        # perform a minibatch update and record the cost for this batch
        Xb = sample_batch(Xtr, bs=batch_size)
        Mb = sample_data_masks(Xb, drop_prob=drop_prob, occ_dim=occ_dim)
        result = SCG.train_joint(Xb, Mb)
        costs = [(costs[j] + result[j]) for j in range(len(result))]
        # output diagnostic information and checkpoint parameters, etc.
        if ((i % 250) == 0):
            costs = [(v / 250.0) for v in costs]
            str1 = "-- batch {0:d} --".format(i)
            str2 = "    total_cost: {0:.4f}".format(costs[0])
            str3 = "    nll_term  : {0:.4f}".format(costs[1])
            str4 = "    kld_q2p   : {0:.4f}".format(costs[2])
            str5 = "    kld_p2q   : {0:.4f}".format(costs[3])
            str6 = "    kld_amu   : {0:.4f}".format(costs[4])
            str7 = "    kld_alv   : {0:.4f}".format(costs[5])
            str8 = "    reg_term  : {0:.4f}".format(costs[6])
            joint_str = "\n".join([str1, str2, str3, str4, str5, str6, str7, str8])
            print(joint_str)
            out_file.write(joint_str+"\n")
            out_file.flush()
            costs = [0.0 for v in costs]
        if ((i % 500) == 0):
            SCG.save_model_params("{}_params.pkl".format(result_tag))
            #############################################
            # check model performance on validation set #
            #############################################
            Xb = sample_batch(Xva, bs=500)
            Mb = sample_data_masks(Xb, drop_prob=drop_prob, occ_dim=occ_dim)
            result = SCG.compute_nll_bound(Xb, Mb)
            str2 = "    va_total_cost: {0:.4f}".format(float(result[0]))
            str3 = "    va_nll_term  : {0:.4f}".format(float(result[1]))
            str4 = "    va_kld_q2p   : {0:.4f}".format(float(result[2]))
            str5 = "    va_kld_p2q   : {0:.4f}".format(float(result[3]))
            str6 = "    va_kld_amu   : {0:.4f}".format(float(result[4]))
            str7 = "    va_kld_alv   : {0:.4f}".format(float(result[5]))
            str8 = "    va_reg_term  : {0:.4f}".format(float(result[6]))
            joint_str = "\n".join([str2, str3, str4, str5, str6, str7, str8])
            print(joint_str)
            out_file.write(joint_str+"\n")
            out_file.flush()
            ###########################################
            # sample and draw attention trajectories. #
            ###########################################
            Xb = sample_batch(Xva, bs=32)
            Mb = sample_data_masks(Xb, drop_prob=drop_prob, occ_dim=occ_dim)
            result = SCG.sample_attention(Xb, Mb)
            post_tag = "b{0:d}".format(i)
            visualize_attention(result, pre_tag=result_tag, post_tag=post_tag)

if __name__=="__main__":
    #test_seq_cond_gen_copy(step_type='add', res_tag="CPY")
    test_seq_cond_gen_impute(step_type='add', res_tag="IMP")
