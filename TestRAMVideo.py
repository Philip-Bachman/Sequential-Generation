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
from HelperFuncs import construct_masked_data, shift_and_scale_into_01, \
                        row_shuffle, to_fX, one_hot_np
from MotionRenderers import TrajectoryGenerator, get_object_painters

RESULT_PATH = "RAM_TEST_RESULTS/"


BREAK_STR = """
#############################################################################
#############################################################################
#############################################################################
#############################################################################
#############################################################################
#############################################################################
#############################################################################
#############################################################################
"""

def test_seq_cond_gen_s(x_objs=['circle'], y_objs=[0], \
                               res_tag="AAA"):
    ##############################
    # File tag, for output stuff #
    ##############################
    result_tag = "{}AAA_VID_SCGS_{}".format(RESULT_PATH, res_tag)

    batch_size = 192
    traj_len = 15
    im_dim = 48
    obs_dim = im_dim*im_dim

    # configure a trajectory generator
    obj_scale = 0.15
    im_box = 1.0 - obj_scale
    x_range = [-im_box,im_box]
    y_range = [-im_box,im_box]
    max_speed = 0.15
    TRAJ = TrajectoryGenerator(x_range=x_range, y_range=y_range, \
                               max_speed=max_speed)
    # configure object renderers for the allowed object types...
    obj_types = ['circle', 'cross', 'square', 't-up', 't-down',
                 't-left', 't-right']
    OPTRS = get_object_painters(im_dim=im_dim, obj_types=obj_types, obj_scale=obj_scale)

    def generate_batch(num_samples, obj_type='circle'):
        # generate a minibatch of trajectories
        traj_pos, traj_vel = TRAJ.generate_trajectories(num_samples, traj_len)
        traj_x = traj_pos[:,:,0]
        traj_y = traj_pos[:,:,1]
        # draw the trajectories
        center_x = to_fX( traj_x.T.ravel() )
        center_y = to_fX( traj_y.T.ravel() )
        delta = to_fX( np.ones(center_x.shape) )
        sigma = to_fX( np.ones(center_x.shape) )
        paint_obj = OPTRS[obj_type]
        W = paint_obj(center_y, center_x, delta, 0.05*sigma)
        # shape trajectories into a batch for passing to the model
        batch_imgs = np.zeros((num_samples, traj_len, obs_dim))
        batch_coords = np.zeros((num_samples, traj_len, 2))
        for i in range(num_samples):
            start_idx = i * traj_len
            end_idx = start_idx + traj_len
            img_set = W[start_idx:end_idx,:]
            batch_imgs[i,:,:] = img_set
            batch_coords[i,:,0] = center_x[start_idx:end_idx]
            batch_coords[i,:,1] = center_y[start_idx:end_idx]
        batch_imgs = np.swapaxes(batch_imgs, 0, 1)
        batch_coords = np.swapaxes(batch_coords, 0, 1)
        return [to_fX( batch_imgs ), to_fX( batch_coords )]

    def generate_batch_multi(num_samples, xobjs=['circle'], yobjs=[0], img_scale=1.0):
        obj_imgs = []
        obj_coords = []
        for obj in xobjs:
            imgs, coords = generate_batch(num_samples, obj_type=obj)
            obj_imgs.append(imgs)
            obj_coords.append(coords)
        seq_len = obj_coords[0].shape[0]
        batch_size = obj_coords[0].shape[1]
        x_imgs = np.zeros(obj_imgs[0].shape)
        y_imgs = np.zeros(obj_imgs[0].shape)
        y_coords = np.zeros(obj_coords[0].shape)
        for o_num in range(len(xobjs)):
            x_imgs = x_imgs + obj_imgs[o_num]
            if o_num in yobjs:
                y_imgs = y_imgs + obj_imgs[o_num]
            mask = npr.rand(seq_len, batch_size) < (1. / (o_num+1))
            mask = mask[:,:,np.newaxis]
            y_coords = (mask * obj_coords[o_num]) + ((1.-mask) * y_coords)
        # rescale coordinates as desired
        y_coords = img_scale * y_coords
        # add noise to image sequences
        pix_mask = npr.rand(*x_imgs.shape) < 0.05
        pix_noise = npr.rand(*x_imgs.shape)
        x_imgs = x_imgs + (pix_mask * pix_noise)
        # clip to 0...0.99
        x_imgs = np.maximum(x_imgs, 0.001)
        x_imgs = np.minimum(x_imgs, 0.999)
        y_imgs = np.maximum(y_imgs, 0.001)
        y_imgs = np.minimum(y_imgs, 0.999)
        return [to_fX(x_imgs), to_fX(y_imgs), to_fX(y_coords)]

    ############################################################
    # Setup some parameters for the Iterative Refinement Model #
    ############################################################
    total_steps = traj_len
    init_steps = 10
    exit_rate = 0.0
    nll_weight = 0.5
    x_dim = obs_dim
    y_dim = obs_dim
    z_att_dim = 128
    z_slf_dim = 128
    att_spec_dim = 5
    glimpse_count = 1
    all_spec_dim = att_spec_dim * glimpse_count
    rnn_dim = 1024
    mlp_dim = 1024

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

    def visualize_attention_joint(result, pre_tag="AAA", post_tag="AAA"):
        seq_len = result[0].shape[0]
        samp_count = result[0].shape[1]
        # get generated predictions
        seq_samps = np.zeros((3*seq_len*samp_count, obs_dim))
        idx = 0
        for s1 in range(samp_count):
            for s2 in range(seq_len):
                seq_samps[idx] = result[3][s2,s1,:]
                idx += 1
            for s2 in range(seq_len):
                seq_samps[idx] = result[0][s2,s1,:]
                idx += 1
            for s2 in range(seq_len):
                seq_samps[idx] = result[1][s2,s1,:]
                idx += 1
        file_name = "{0:s}_traj_joint_{1:s}.png".format(pre_tag, post_tag)
        utils.visualize_samples(seq_samps, file_name, num_rows=(3*samp_count))
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
                                      img_scale=img_scale, att_scale=0.33,
                                      **inits)
    glimpse_dim = reader_mlp.read_dim # total number of "pixels" read by reader
    all_read_dim = glimpse_count * glimpse_dim

    # MLP for updating belief state based on con_rnn
    writer_mlp = MLP([None, None], [rnn_dim, mlp_dim, obs_dim], \
                     name="writer_mlp", **inits)

    # mlps for processing inputs to attention module
    att_mlp_in = CondNet([Rectifier()], [z_att_dim, mlp_dim, all_spec_dim], \
                          name="att_mlp_in", **inits)

    # mlps for processing inputs to LSTMs
    con_mlp_in = MLP([Identity()], \
                     [(z_slf_dim + all_spec_dim + all_read_dim), 4*rnn_dim], \
                     name="con_mlp_in", **inits)
    rav_mlp_in = MLP([Identity()], \
                     [(all_read_dim + rnn_dim + z_slf_dim + all_spec_dim + all_read_dim), 4*rnn_dim], \
                     name="rav_mlp_in", **inits)

    # mlps for turning LSTM outputs into conditionals over z_gen
    con_mlp_out = CondNet([], [rnn_dim, (z_att_dim + z_slf_dim)], \
                          name="con_mlp_out", **inits)
    rav_mlp_out = CondNet([], [rnn_dim, (z_att_dim + z_slf_dim)], \
                          name="rav_mlp_out", **inits)

    # LSTMs for the actual LSTMs (obviously, perhaps)
    con_rnn = BiasedLSTM(dim=rnn_dim, ig_bias=2.0, fg_bias=1.0, \
                         name="con_rnn", **rnninits)
    rav_rnn = BiasedLSTM(dim=rnn_dim, ig_bias=2.0, fg_bias=1.0, \
                         name="rav_rnn", **rnninits)

    SeqCondGen2dS_doc_str = \
    """
    SeqCondGen2dS -- develops beliefs subject to constraints on perception.

    This model sequentially constructs a conditional density estimate by taking
    repeated glimpses at the input x, while formulating a hypothesis about the
    output y. The objective is maximum likelihood for (x,y) pairs drawn from
    some training set.

    The input pairs (x, y) can be either "static" or "sequential". In the
    static case, the same x and y are used at every step of the hypothesis
    construction loop. In the sequential case, x and y can change at each step
    of the loop.

    ***                                                                     ***
    *** This version of the model assumes the use of a 2d attention module. ***
    ***                                                                     ***
    *** The attention module should accept 5d inputs for specifying the     ***
    *** location, scale, etc. of the attention-based reader.                ***
    ***                                                                     ***
    *** Multiple glimpses can be taken in each processing step.             ***
    ***                                                                     ***

    Parameters:
        x_and_y_are_seqs: boolean telling whether the conditioning information
                          and prediction targets are sequential.
        total_steps: total number of steps in sequential estimation process
        init_steps: number of steps prior to first NLL measurement
        exit_rate: probability of exiting following each non "init" step
                   **^^ THIS IS SET TO 0 WHEN USING SEQUENTIAL INPUT ^^**
        nll_weight: weight for the prediction NLL term at each step.
                   **^^ THIS IS IGNORED WHEN USING STATIC INPUT ^^**
        x_dim: dimension of inputs on which to condition
        y_dim: dimension of outputs to predict
        att_spec_dim: dimension of the specification for each glimpse
        glimpse_dim: dimension of the read-out from each glimpse
        glimpse_count: number of glimpses to take per step. the input and output
                       dimensions of the relevant child networks should be set
                       up to provide glimpse_count glimpse locations, and to
                       receive glimpse_count attention read-outs.
        reader_mlp: used for reading from the input
                    -- this is a 2d attention module for which the attention
                       location is specified by 5 inputs, the first two of
                       which we will assume are (x,y) coordinates.
        writer_mlp: used for writing to the output prediction
                    -- for "add" steps, this takes the controller's visible
                       state as input and updates the current "belief" state.
                       for "jump" steps, this converts the controller's memory
                       state into the current "belief" state.
        con_mlp_in: preprocesses input to the "controller" LSTM
        con_rnn: the "controller" LSTM
        con_mlp_out: CondNet for distribution over z given con_rnn
        rav_mlp_in: preprocesses input to the "variational controller" LSTM
        rav_rnn: the "variational controller" LSTM
        rav_mlp_out: CondNet for distribution over z given rav_rnn
        att_mlp_in: CondNet to convert a subset of z into attention specs
    """

    SCG = SeqCondGen2dS(
                x_and_y_are_seqs=True,
                total_steps=total_steps,
                init_steps=init_steps,
                exit_rate=exit_rate,
                nll_weight=nll_weight,
                x_dim=obs_dim,
                y_dim=obs_dim,
                att_spec_dim=att_spec_dim,
                glimpse_dim=glimpse_dim,
                glimpse_count=glimpse_count,
                reader_mlp=reader_mlp,
                writer_mlp=writer_mlp,
                con_mlp_in=con_mlp_in,
                con_mlp_out=con_mlp_out,
                con_rnn=con_rnn,
                rav_mlp_in=rav_mlp_in,
                rav_mlp_out=rav_mlp_out,
                rav_rnn=rav_rnn,
                att_mlp_in=att_mlp_in)
    SCG.initialize()

    compile_start_time = time.time()

    # build the attention trajectory sampler
    SCG.build_attention_funcs()

    # quick test of attention trajectory sampler
    samp_count = 32
    Xb, Yb, Cb = generate_batch_multi(samp_count, xobjs=x_objs, yobjs=y_objs, img_scale=img_scale)
    result = SCG.sample_attention(Xb, Yb)
    visualize_attention_joint(result, pre_tag=result_tag, post_tag="b0")

    # build the main model functions (i.e. training and cost functions)
    SCG.build_model_funcs()

    compile_end_time = time.time()
    compile_minutes = (compile_end_time - compile_start_time) / 60.0
    print("THEANO COMPILE TIME (MIN): {}".format(compile_minutes))

    # TEST SAVE/LOAD FUNCTIONALITY
    param_save_file = "{}_params.pkl".format(result_tag)
    #SCG.save_model_params(param_save_file)
    #SCG.load_model_params(param_save_file)

    ################################################################
    # Apply some updates, to check that they aren't totally broken #
    ################################################################
    print("Beginning to train the model...")
    out_file = open("{}_results.txt".format(result_tag), 'wb')
    out_file.flush()
    costs = [0. for i in range(10)]
    learn_rate = 0.0001
    momentum = 0.95
    kl_scale = 1.0
    for i in range(500000):
        scale = min(1.0, ((i+1) / 5000.0))
        if (((i + 1) % 10000) == 0):
            learn_rate = learn_rate * 0.96
        if ((i > 200000) and (i % 10000)):
            kl_scale = kl_scale * 1.1
        # set sgd and objective function hyperparams for this update
        SCG.set_sgd_params(lr=scale*learn_rate, mom_1=scale*momentum, mom_2=0.99)
        SCG.set_lam_kld(lam_kld_q2p=kl_scale*2.0, lam_kld_p2q=kl_scale*0.2)
        # perform a minibatch update and record the cost for this batch
        Xb, Yb, Cb = generate_batch_multi(samp_count, xobjs=x_objs, yobjs=y_objs, img_scale=img_scale)
        result = SCG.train_joint(Xb, Yb)
        costs = [(costs[j] + result[j]) for j in range(len(result))]
        # output diagnostic information and checkpoint parameters, etc.
        if ((i % 1000) == 0):
            costs = [(v / 1000.0) for v in costs]
            str1 = "-- batch {0:d} --".format(i)
            str2 = "    total_cost: {0:.4f}".format(costs[0])
            str3 = "    nll_term  : {0:.4f}".format(costs[1])
            str4 = "    kld_q2p   : {0:.4f}".format(costs[2])
            str5 = "    kld_p2q   : {0:.4f}".format(costs[3])
            str6 = "    reg_term  : {0:.4f}".format(costs[4])
            joint_str = "\n".join([str1, str2, str3, str4, str5, str6])
            print(joint_str)
            out_file.write(joint_str+"\n")
            out_file.flush()
            costs = [0.0 for v in costs]
        if ((i % 5000) == 0):
            SCG.save_model_params("{}_params.pkl".format(result_tag))
            ###########################################
            # Sample and draw attention trajectories. #
            ###########################################
            samp_count = 32
            Xb, Yb, Cb = generate_batch_multi(samp_count, xobjs=x_objs, yobjs=y_objs, img_scale=img_scale)
            result = SCG.sample_attention(Xb, Yb)
            post_tag = "b{0:d}".format(i)
            visualize_attention_joint(result, pre_tag=result_tag, post_tag=post_tag)


BREAK_STR = """
#############################################################################
#############################################################################
#############################################################################
#############################################################################
#############################################################################
#############################################################################
#############################################################################
#############################################################################
"""


def test_seq_cond_gen_ram(x_objs=['circle'], y_objs=[0], \
                          res_tag="AAA"):
    ##############################
    # File tag, for output stuff #
    ##############################
    result_tag = "{}AAA_VID_SCGRAM_{}".format(RESULT_PATH, res_tag)

    batch_size = 192
    traj_len = 15
    im_dim = 48
    obs_dim = im_dim*im_dim

    # configure a trajectory generator
    obj_scale = 0.15
    im_box = 1.0 - obj_scale
    x_range = [-im_box,im_box]
    y_range = [-im_box,im_box]
    max_speed = 0.15
    TRAJ = TrajectoryGenerator(x_range=x_range, y_range=y_range, \
                               max_speed=max_speed)
    # configure object renderers for the allowed object types...
    obj_types = ['circle', 'cross', 'square', 't-up', 't-down',
                 't-left', 't-right']
    OPTRS = get_object_painters(im_dim=im_dim, obj_types=obj_types, obj_scale=obj_scale)

    def generate_batch(num_samples, obj_type='circle'):
        # generate a minibatch of trajectories
        traj_pos, traj_vel = TRAJ.generate_trajectories(num_samples, traj_len)
        traj_x = traj_pos[:,:,0]
        traj_y = traj_pos[:,:,1]
        # draw the trajectories
        center_x = to_fX( traj_x.T.ravel() )
        center_y = to_fX( traj_y.T.ravel() )
        delta = to_fX( np.ones(center_x.shape) )
        sigma = to_fX( np.ones(center_x.shape) )
        paint_obj = OPTRS[obj_type]
        W = paint_obj(center_y, center_x, delta, 0.05*sigma)
        # shape trajectories into a batch for passing to the model
        batch_imgs = np.zeros((num_samples, traj_len, obs_dim))
        batch_coords = np.zeros((num_samples, traj_len, 2))
        for i in range(num_samples):
            start_idx = i * traj_len
            end_idx = start_idx + traj_len
            img_set = W[start_idx:end_idx,:]
            batch_imgs[i,:,:] = img_set
            batch_coords[i,:,0] = center_x[start_idx:end_idx]
            batch_coords[i,:,1] = center_y[start_idx:end_idx]
        batch_imgs = np.swapaxes(batch_imgs, 0, 1)
        batch_coords = np.swapaxes(batch_coords, 0, 1)
        return [to_fX( batch_imgs ), to_fX( batch_coords )]

    def generate_batch_multi(num_samples, xobjs=['circle'], yobjs=[0], img_scale=1.0):
        obj_imgs = []
        obj_coords = []
        for obj in xobjs:
            imgs, coords = generate_batch(num_samples, obj_type=obj)
            obj_imgs.append(imgs)
            obj_coords.append(coords)
        seq_len = obj_coords[0].shape[0]
        batch_size = obj_coords[0].shape[1]
        x_imgs = np.zeros(obj_imgs[0].shape)
        y_imgs = np.zeros(obj_imgs[0].shape)
        y_coords = np.zeros(obj_coords[0].shape)
        for o_num in range(len(xobjs)):
            x_imgs = x_imgs + obj_imgs[o_num]
            if o_num in yobjs:
                y_imgs = y_imgs + obj_imgs[o_num]
            mask = npr.rand(seq_len, batch_size) < (1. / (o_num+1))
            mask = mask[:,:,np.newaxis]
            y_coords = (mask * obj_coords[o_num]) + ((1.-mask) * y_coords)
        # rescale coordinates as desired
        y_coords = img_scale * y_coords
        # add noise to image sequences
        pix_mask = npr.rand(*x_imgs.shape) < 0.05
        pix_noise = npr.rand(*x_imgs.shape)
        x_imgs = x_imgs + (pix_mask * pix_noise)
        # clip to 0...0.99
        x_imgs = np.maximum(x_imgs, 0.001)
        x_imgs = np.minimum(x_imgs, 0.999)
        y_imgs = np.maximum(y_imgs, 0.001)
        y_imgs = np.minimum(y_imgs, 0.999)
        return [to_fX(x_imgs), to_fX(y_imgs), to_fX(y_coords)]

    ############################################################
    # Setup some parameters for the Iterative Refinement Model #
    ############################################################
    total_steps = traj_len
    init_steps = 10
    exit_rate = 0.0
    nll_weight = 0.5
    x_dim = obs_dim
    y_dim = obs_dim
    z_dim = 256
    att_spec_dim = 5
    rnn_dim = 1024
    mlp_dim = 1024

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

    def visualize_attention_joint(result, pre_tag="AAA", post_tag="AAA"):
        seq_len = result[0].shape[0]
        samp_count = result[0].shape[1]
        # get generated predictions
        seq_samps = np.zeros((3*seq_len*samp_count, obs_dim))
        idx = 0
        for s1 in range(samp_count):
            for s2 in range(seq_len):
                seq_samps[idx] = result[3][s2,s1,:]
                idx += 1
            for s2 in range(seq_len):
                seq_samps[idx] = result[0][s2,s1,:]
                idx += 1
            for s2 in range(seq_len):
                seq_samps[idx] = result[1][s2,s1,:]
                idx += 1
        file_name = "{0:s}_traj_joint_{1:s}.png".format(pre_tag, post_tag)
        utils.visualize_samples(seq_samps, file_name, num_rows=(3*samp_count))
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
                                      img_scale=img_scale, att_scale=0.33,
                                      **inits)
    read_dim = reader_mlp.read_dim # total number of "pixels" read by reader

    # MLP for updating belief state based on con_rnn
    writer_mlp = MLP([None, None], [rnn_dim, mlp_dim, obs_dim], \
                     name="writer_mlp", **inits)

    # mlps for processing inputs to LSTMs
    con_mlp_in = MLP([Identity()], \
                     [(z_dim + rnn_dim), 4*rnn_dim], \
                     name="con_mlp_in", **inits)
    var_mlp_in = MLP([Identity()], \
                     [(read_dim + read_dim + att_spec_dim + rnn_dim), 4*rnn_dim], \
                     name="var_mlp_in", **inits)
    gen_mlp_in = MLP([Identity()], \
                     [(read_dim + att_spec_dim + rnn_dim), 4*rnn_dim], \
                     name="gen_mlp_in", **inits)

    # mlps for turning LSTM outputs into conditionals over z_gen
    con_mlp_out = CondNet([Rectifier()], [rnn_dim, mlp_dim, att_spec_dim], \
                          name="con_mlp_out", **inits)
    gen_mlp_out = CondNet([], [rnn_dim, z_dim], name="gen_mlp_out", **inits)
    var_mlp_out = CondNet([], [rnn_dim, z_dim], name="var_mlp_out", **inits)

    # LSTMs for the actual LSTMs (obviously, perhaps)
    con_rnn = BiasedLSTM(dim=rnn_dim, ig_bias=2.0, fg_bias=1.0, \
                         name="con_rnn", **rnninits)
    gen_rnn = BiasedLSTM(dim=rnn_dim, ig_bias=2.0, fg_bias=1.0, \
                         name="gen_rnn", **rnninits)
    var_rnn = BiasedLSTM(dim=rnn_dim, ig_bias=2.0, fg_bias=1.0, \
                         name="var_rnn", **rnninits)

    SeqCondGenRAM_doc_str = \
    """
    SeqCondGenRAM -- constructs conditional densities under time constraints.

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
        x_dim: dimension of inputs on which to condition
        y_dim: dimension of outputs to predict
        reader_mlp: used for reading from the input
        writer_mlp: used for writing to the output prediction
        con_mlp_in: preprocesses input to the "controller" LSTM
        con_rnn: the "controller" LSTM
        con_mlp_out: CondNet for distribution over att spec given con_rnn
        gen_mlp_in: preprocesses input to the "generator" LSTM
        gen_rnn: the "generator" LSTM
        gen_mlp_out: CondNet for distribution over z given gen_rnn
        var_mlp_in: preprocesses input to the "variational" LSTM
        var_rnn: the "variational" LSTM
        var_mlp_out: CondNet for distribution over z given gen_rnn
    """

    SCG = SeqCondGenRAM(
                x_and_y_are_seqs=True,
                total_steps=total_steps,
                init_steps=init_steps,
                exit_rate=exit_rate,
                nll_weight=nll_weight,
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
                att_noise=0.02)
    SCG.initialize()

    compile_start_time = time.time()

    # build the attention trajectory sampler
    SCG.build_attention_funcs()

    # TEST SAVE/LOAD FUNCTIONALITY
    param_save_file = "{}_params.pkl".format(result_tag)
    #SCG.save_model_params(param_save_file)
    #SCG.load_model_params(param_save_file)

    # quick test of attention trajectory sampler
    samp_count = 32
    Xb, Yb, Cb = generate_batch_multi(samp_count, xobjs=x_objs, yobjs=y_objs, img_scale=img_scale)
    result = SCG.sample_attention(Xb, Yb)
    visualize_attention_joint(result, pre_tag=result_tag, post_tag="b0")

    # build the main model functions (i.e. training and cost functions)
    SCG.build_model_funcs()

    compile_end_time = time.time()
    compile_minutes = (compile_end_time - compile_start_time) / 60.0
    print("THEANO COMPILE TIME (MIN): {}".format(compile_minutes))

    ################################################################
    # Apply some updates, to check that they aren't totally broken #
    ################################################################
    print("Beginning to train the model...")
    out_file = open("{}_results.txt".format(result_tag), 'wb')
    out_file.flush()
    costs = [0. for i in range(10)]
    learn_rate = 0.0001
    momentum = 0.95
    kl_scale = 1.0
    for i in range(500000):
        scale = min(1.0, ((i+1) / 5000.0))
        if (((i + 1) % 10000) == 0):
            learn_rate = learn_rate * 0.96
        if ((i > 200000) and (i % 10000)):
            kl_scale = kl_scale * 1.1
        # set sgd and objective function hyperparams for this update
        SCG.set_sgd_params(lr=scale*learn_rate, mom_1=scale*momentum, mom_2=0.99)
        SCG.set_lam_kld(lam_kld_q2p=kl_scale*2.0, lam_kld_p2q=kl_scale*0.2, \
                        lam_kld_amu=0.0, lam_kld_alv=0.1)
        # perform a minibatch update and record the cost for this batch
        Xb, Yb, Cb = generate_batch_multi(samp_count, xobjs=x_objs, yobjs=y_objs, img_scale=img_scale)
        result = SCG.train_joint(Xb, Yb)
        costs = [(costs[j] + result[j]) for j in range(len(result))]
        # output diagnostic information and checkpoint parameters, etc.
        if ((i % 1000) == 0):
            costs = [(v / 1000.0) for v in costs]
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
        if ((i % 5000) == 0):
            SCG.save_model_params("{}_params.pkl".format(result_tag))
            ###########################################
            # Sample and draw attention trajectories. #
            ###########################################
            samp_count = 32
            Xb, Yb, Cb = generate_batch_multi(samp_count, xobjs=x_objs, yobjs=y_objs, img_scale=img_scale)
            result = SCG.sample_attention(Xb, Yb)
            post_tag = "b{0:d}".format(i)
            #visualize_attention(result, pre_tag=result_tag, post_tag=post_tag)
            visualize_attention_joint(result, pre_tag=result_tag, post_tag=post_tag)

if __name__=="__main__":
    # Test SeqCondGenX
    #test_seq_cond_gen_x(x_objs=['cross', 'circle', 'circle'], y_objs=[0], res_tag="T1")
    #test_seq_cond_gen_x(x_objs=['cross', 'circle'], y_objs=[0,1], res_tag="T2")
    #test_seq_cond_gen_x(x_objs=['t-up', 't-down', 'circle'], y_objs=[0,1], res_tag="T3")
    # Test SeqCondGenS
    #test_seq_cond_gen_s(x_objs=['cross', 'circle', 'circle'], y_objs=[0], res_tag="T1")
    #test_seq_cond_gen_s(x_objs=['cross', 'circle'], y_objs=[0,1], res_tag="T2")
    test_seq_cond_gen_s(x_objs=['t-up', 't-down', 'circle'], y_objs=[0,1], res_tag="T3")
    # Test SeqCondGenRAM
    #test_seq_cond_gen_ram(x_objs=['cross', 'circle', 'circle'], y_objs=[0], res_tag="T1")
    #test_seq_cond_gen_ram(x_objs=['cross', 'circle'], y_objs=[0,1], res_tag="T2")
    #test_seq_cond_gen_ram(x_objs=['t-up', 't-down', 'circle'], y_objs=[0,1], res_tag="T3")
