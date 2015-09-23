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
from MotionRenderers import TrajectoryGenerator, ObjectPainter

RESULT_PATH = "RAM_TEST_RESULTS/"

def test_seq_cond_gen_sequence(step_type='add'):
    ##############################
    # File tag, for output stuff #
    ##############################
    result_tag = "{}VID_SCG".format(RESULT_PATH)

    batch_size = 100
    traj_len = 64
    im_dim = 32
    obs_dim = im_dim*im_dim

    # configure a trajectory generator
    x_range = [-0.8,0.8]
    y_range = [-0.8,0.8]
    max_speed = 0.15
    TRAJ = TrajectoryGenerator(x_range=x_range, y_range=y_range, \
                               max_speed=max_speed)
    # configure an object renderer
    OPTR = ObjectPainter(im_dim, im_dim, obj_type='circle', obj_scale=0.2)
    # get a Theano function for doing the rendering
    _center_x = T.vector()
    _center_y = T.vector()
    _delta = T.vector()
    _sigma = T.vector()
    _W = OPTR.write(_center_y, _center_x, _delta, _sigma)
    write_func = theano.function(inputs=[_center_y, _center_x, _delta, _sigma], \
                                 outputs=_W)

    def generate_batch(num_samples):
        # generate a minibatch of trajectories
        traj_pos, traj_vel = TRAJ.generate_trajectories(num_samples, traj_len)
        traj_x = traj_pos[:,:,0]
        traj_y = traj_pos[:,:,1]
        # draw the trajectories
        center_x = to_fX( traj_x.T.ravel() )
        center_y = to_fX( traj_y.T.ravel() )
        delta = to_fX( np.ones(center_x.shape) )
        sigma = to_fX( np.ones(center_x.shape) )
        W = write_func(center_y, center_x, delta, 0.05*sigma)
        # shape trajectories into a batch for passing to the model
        batch_imgs = np.zeros((num_samples, traj_len, obs_dim))
        for i in range(num_samples):
            start_idx = i * traj_len
            end_idx = start_idx + traj_len
            img_set = W[start_idx:end_idx,:]
            batch_imgs[i,:,:] = img_set
        batch_imgs = np.swapaxes(batch_imgs, 0, 1)
        batch_imgs = to_fX( batch_imgs )
        return batch_imgs

    ############################################################
    # Setup some parameters for the Iterative Refinement Model #
    ############################################################
    total_steps = traj_len
    init_steps = 3
    exit_rate = 0.0
    nll_weight = 0.2
    x_dim = obs_dim
    y_dim = obs_dim
    z_dim = 100
    rnn_dim = 400
    write_dim = 400
    mlp_dim = 400

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
                seq_samps[idx] = result[1][s2,s1,:obs_dim] + result[1][s2,s1,obs_dim:]
                idx += 1
        file_name = "{0:s}_traj_att_maps_{1:s}.png".format(pre_tag, post_tag)
        utils.visualize_samples(seq_samps, file_name, num_rows=samp_count)
        # get sequential attention maps (read out values)
        seq_samps = np.zeros((seq_len*samp_count, obs_dim))
        idx = 0
        for s1 in range(samp_count):
            for s2 in range(seq_len):
                seq_samps[idx] = result[2][s2,s1,:obs_dim] + result[2][s2,s1,obs_dim:]
                idx += 1
        file_name = "{0:s}_traj_read_outs_{1:s}.png".format(pre_tag, post_tag)
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

    read_N = 2 # inner/outer grid dimension for reader
    reader_mlp = SimpleAttentionReader2d(x_dim=obs_dim, con_dim=rnn_dim,
                                         width=im_dim, height=im_dim, N=read_N,
                                         img_scale=0.2, att_scale=0.5,
                                         **inits)
    read_dim = reader_mlp.read_dim # total number of "pixels" read by reader

    writer_mlp = MLP([None, None], [rnn_dim, write_dim, obs_dim], \
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
                x_and_y_are_seqs=True,
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
    samp_count = 32
    Xb = generate_batch(samp_count)
    result = SCG.sample_attention(Xb, Xb)
    result[0] = Xb
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
    momentum = 0.9
    for i in range(250000):
        scale = min(1.0, ((i+1) / 2000.0))
        if (((i + 1) % 10000) == 0):
            learn_rate = learn_rate * 0.95
        if (i > 5000):
            momentum = 0.95
        else:
            momentum = 0.9
        # set sgd and objective function hyperparams for this update
        SCG.set_sgd_params(lr=learn_rate, mom_1=momentum, mom_2=0.99)
        SCG.set_lam_kld(lam_kld_q2p=0.95, lam_kld_p2q=0.05, lam_kld_p2g=0.0)
        # perform a minibatch update and record the cost for this batch
        Xb = generate_batch(batch_size)
        result = SCG.train_joint(Xb, Xb)
        costs = [(costs[j] + result[j]) for j in range(len(result))]

        # output diagnostic information and checkpoint parameters, etc.
        if ((i % 100) == 0):
            costs = [(v / 100.0) for v in costs]
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
        if ((i % 200) == 0):
            SCG.save_model_params("{}_params.pkl".format(result_tag))
            # compute a small-sample estimate of NLL bound on validation set
            samp_count = 128
            Xb = generate_batch(samp_count)
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
            samp_count = 32
            Xb = generate_batch(samp_count)
            result = SCG.sample_attention(Xb, Xb)
            post_tag = "b{0:d}".format(i)
            visualize_attention(result, pre_tag=result_tag, post_tag=post_tag)



if __name__=="__main__":
    test_seq_cond_gen_sequence(step_type='add')
