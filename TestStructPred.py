##################################################################
# Code for testing the variational Multi-Stage Generative Model. #
##################################################################

from __future__ import print_function, division

# basic python
import time
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
from blocks.model import Model
from blocks.bricks import Tanh, Identity, Rectifier
from blocks.bricks.recurrent import SimpleRecurrent, LSTM

# phil's sweetness
import utils
from BlocksModels import *
from load_data import load_udm, load_mnist, load_binarized_mnist
from HelperFuncs import construct_masked_data, shift_and_scale_into_01, \
                        row_shuffle, to_fX, binarize_data

################################
################################
## HELPER FUNCTIONS FOR STUFF ##
################################
################################

def img_split(imgs, im_dim=None, split_col=None, transposed=False):
    """
    Split flattened images in rows of img vertically, with obs_cols taken
    from the left and im_dim[1]-obs_cols taken from the right.
    """
    if transposed:
        assert (im_dim[0] == im_dim[1]), "transpose only works for square imgs"
    img_count = imgs.shape[0]
    row_count = im_dim[0]
    col_count = im_dim[1]
    l_obs_dim = split_col * row_count
    r_obs_dim = (col_count - split_col) * row_count
    left_cols = np.zeros((img_count, l_obs_dim))
    right_cols = np.zeros((img_count, r_obs_dim))
    for i in range(img_count):
        im = imgs[i,:].reshape(im_dim)
        if transposed:
            im = im.transpose()
        left_cols[i,:] = im[:,:split_col].flatten()
        right_cols[i,:] = im[:,split_col:].flatten()
    return to_fX(left_cols), to_fX(right_cols)

def img_join(left_cols, right_cols, im_dim=None, transposed=False):
    """
    Join flattened images vertically.
    """
    if transposed:
        assert (im_dim[0] == im_dim[1]), "transpose only works for square imgs"
    img_count = left_cols.shape[0]
    row_count = im_dim[0]
    col_count = im_dim[1]
    left_col_count = left_cols.shape[1] / row_count
    right_col_count = col_count - left_col_count
    imgs = np.zeros((img_count, row_count*col_count))
    im_sq = np.zeros((row_count, col_count))
    for i in range(img_count):
        left_chunk = left_cols[i,:].reshape((row_count, left_col_count))
        right_chunk = right_cols[i,:].reshape((row_count, right_col_count))
        im_sq[:,:left_col_count] = left_chunk[:,:]
        im_sq[:,left_col_count:] = right_chunk[:,:]
        if transposed:
            im_sq = im_sq.transpose()
        imgs[i,:] = im_sq.flatten()
    return to_fX(imgs)

def seq_img_join(left_seq, right_seq, im_dim=None, transposed=False):
    """
    Join a sequence of images that were split vertically.
    """
    if transposed:
        assert (im_dim[0] == im_dim[1]), "transpose only works for square imgs"
    img_seq = [img_join(left_seq[i,:,:], right_seq[i,:,:], im_dim=im_dim, transposed=transposed) \
               for i in range(left_seq.shape[0])]
    return img_seq


#############################
#############################
## TEST BASIC RLDRAW MODEL ##
#############################
#############################

def test_lstm_structpred(step_type='add', use_pol=True, use_binary=False):
    ###########################################
    # Make a tag for identifying result files #
    ###########################################
    pol_tag = "P1" if use_pol else "P0"
    bin_tag = "B1" if use_binary else "B0"
    res_tag = "SP_LSTM_{}_{}_{}".format(step_type, pol_tag, bin_tag)

    if use_binary:
        ############################
        # Get binary training data #
        ############################
        rng = np.random.RandomState(1234)
        Xtr, Xva, Xte = load_binarized_mnist(data_path='./data/')
        #Xtr = np.vstack((Xtr, Xva))
        #Xva = Xte
    else:
        ################################
        # Get continuous training data #
        ################################
        rng = np.random.RandomState(1234)
        dataset = 'data/mnist.pkl.gz'
        datasets = load_udm(dataset, as_shared=False, zero_mean=False)
        Xtr = datasets[0][0]
        Xva = datasets[1][0]
        Xte = datasets[2][0]
        #Xtr = np.concatenate((Xtr, Xva), axis=0)
        #Xva = Xte
        Xtr = to_fX(shift_and_scale_into_01(Xtr))
        Xva = to_fX(shift_and_scale_into_01(Xva))
    tr_samples = Xtr.shape[0]
    va_samples = Xva.shape[0]
    batch_size = 200


    ########################################################
    # Split data into "observation" and "prediction" parts #
    ########################################################
    obs_cols = 14             # number of columns to observe
    pred_cols = 28 - obs_cols # number of columns to predict
    x_dim = obs_cols * 28     # dimensionality of observations
    y_dim = pred_cols * 28    # dimensionality of predictions
    Xtr, Ytr = img_split(Xtr, im_dim=(28, 28), split_col=obs_cols, transposed=True)
    Xva, Yva = img_split(Xva, im_dim=(28, 28), split_col=obs_cols, transposed=True)

    ############################################################
    # Setup some parameters for the Iterative Refinement Model #
    ############################################################
    read_dim = 128
    write_dim = 128
    mlp_dim = 128
    rnn_dim = 128
    z_dim = 64
    n_iter = 15

    rnninits = {
        'weights_init': IsotropicGaussian(0.01),
        'biases_init': Constant(0.),
    }
    inits = {
        'weights_init': IsotropicGaussian(0.01),
        'biases_init': Constant(0.),
    }

    # setup reader/writer models
    reader_mlp = MLP([Rectifier(), Tanh()], [x_dim, mlp_dim, read_dim],
                     name="reader_mlp", **inits)
    writer_mlp = MLP([Rectifier(), None], [rnn_dim, mlp_dim, y_dim],
                     name="writer_mlp", **inits)

    # setup submodels for processing LSTM inputs
    pol_inp_dim = y_dim + read_dim + rnn_dim
    var_inp_dim = y_dim + y_dim + read_dim + rnn_dim
    pol_mlp_in = MLP([Identity()], [pol_inp_dim, 4*rnn_dim],
                     name="pol_mlp_in", **inits)
    var_mlp_in = MLP([Identity()], [var_inp_dim, 4*rnn_dim],
                     name="var_mlp_in", **inits)
    dec_mlp_in = MLP([Identity()], [z_dim, 4*rnn_dim],
                     name="dec_mlp_in", **inits)

    # setup submodels for turning LSTM states into conditionals over z
    pol_mlp_out = CondNet([], [rnn_dim, z_dim], name="pol_mlp_out", **inits)
    var_mlp_out = CondNet([], [rnn_dim, z_dim], name="var_mlp_out", **inits)
    dec_mlp_out = CondNet([], [rnn_dim, z_dim], name="dec_mlp_out", **inits)

    # setup the LSTMs for primary policy, guide policy, and shared dynamics
    pol_rnn = BiasedLSTM(dim=rnn_dim, ig_bias=2.0, fg_bias=2.0, \
                         name="pol_rnn", **rnninits)
    var_rnn = BiasedLSTM(dim=rnn_dim, ig_bias=2.0, fg_bias=2.0, \
                         name="var_rnn", **rnninits)
    dec_rnn = BiasedLSTM(dim=rnn_dim, ig_bias=2.0, fg_bias=2.0, \
                         name="dec_rnn", **rnninits)

    model = IRStructPredModel(
                n_iter,
                step_type=step_type,
                use_pol=use_pol,
                reader_mlp=reader_mlp,
                writer_mlp=writer_mlp,
                pol_mlp_in=pol_mlp_in,
                pol_mlp_out=pol_mlp_out,
                pol_rnn=pol_rnn,
                var_mlp_in=var_mlp_in,
                var_mlp_out=var_mlp_out,
                var_rnn=var_rnn,
                dec_mlp_in=dec_mlp_in,
                dec_mlp_out=dec_mlp_out,
                dec_rnn=dec_rnn)
    model.initialize()

    compile_start_time = time.time()

    # build the cost gradients, training function, samplers, etc.
    model.build_sampling_funcs()
    print("Testing model sampler...")
    # draw some independent samples from the model
    samp_count = 10
    samp_reps = 3
    x_in = Xtr[:10,:].repeat(samp_reps, axis=0)
    y_in = Ytr[:10,:].repeat(samp_reps, axis=0)
    x_samps, y_samps = model.sample_model(x_in, y_in, sample_source='p')
    # TODO: visualize sample prediction trajectories
    img_seq = seq_img_join(x_samps, y_samps, im_dim=(28,28), transposed=True)
    seq_len = len(img_seq)
    samp_count = img_seq[0].shape[0]
    seq_samps = np.zeros((seq_len*samp_count, img_seq[0].shape[1]))
    idx = 0
    for s1 in range(samp_count):
        for s2 in range(seq_len):
            seq_samps[idx] = img_seq[s2][s1]
            idx += 1
    file_name = "{0:s}_samples_b{1:d}.png".format(res_tag, 0)
    utils.visualize_samples(seq_samps, file_name, num_rows=samp_count)

    model.build_model_funcs()

    compile_end_time = time.time()
    compile_minutes = (compile_end_time - compile_start_time) / 60.0
    print("THEANO COMPILE TIME (MIN): {}".format(compile_minutes))

    ################################################################
    # Apply some updates, to check that they aren't totally broken #
    ################################################################
    print("Beginning to train the model...")
    out_file = open("{}_results.txt".format(res_tag), 'wb')
    costs = [0. for i in range(10)]
    learn_rate = 0.0002
    momentum = 0.9
    batch_idx = np.arange(batch_size) + tr_samples
    for i in range(300000):
        scale = min(1.0, ((i+1) / 5000.0))
        if (((i + 1) % 10000) == 0):
            learn_rate = learn_rate * 0.95
        # get the indices of training samples for this batch update
        batch_idx += batch_size
        if (np.max(batch_idx) >= tr_samples):
            # we finished an "epoch", so we rejumble the training set
            Xtr, Ytr = row_shuffle(Xtr, Ytr)
            batch_idx = np.arange(batch_size)
        # set sgd and objective function hyperparams for this update
        model.set_sgd_params(lr=scale*learn_rate, mom_1=scale*momentum, mom_2=0.98)
        model.set_lam_kld(lam_kld_q2p=1.0, lam_kld_p2q=0.1)
        model.set_grad_noise(grad_noise=0.02)
        # perform a minibatch update and record the cost for this batch
        Xb = to_fX(Xtr.take(batch_idx, axis=0))
        Yb = to_fX(Ytr.take(batch_idx, axis=0))
        result = model.train_joint(Xb, Yb)
        costs = [(costs[j] + result[j]) for j in range(len(result))]

        # diagnostics
        if ((i % 250) == 0):
            costs = [(v / 250.0) for v in costs]
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
            model.save_model_params("{}_params.pkl".format(res_tag))
            # compute a small-sample estimate of NLL bound on validation set
            Xva, Yva = row_shuffle(Xva, Yva)
            Xb = to_fX(Xva[:5000])
            Yb = to_fX(Yva[:5000])
            va_costs = model.compute_nll_bound(Xb, Yb)
            str1 = "    va_nll_bound : {}".format(va_costs[1])
            str2 = "    va_nll_term  : {}".format(va_costs[2])
            str3 = "    va_kld_q2p   : {}".format(va_costs[3])
            joint_str = "\n".join([str1, str2, str3])
            print(joint_str)
            out_file.write(joint_str+"\n")
            out_file.flush()
            # draw some independent samples from the model
            samp_count = 10
            samp_reps = 3
            x_in = Xva[:samp_count,:].repeat(samp_reps, axis=0)
            y_in = Yva[:samp_count,:].repeat(samp_reps, axis=0)
            x_samps, y_samps = model.sample_model(x_in, y_in, sample_source='p')
            # visualize sample prediction trajectories
            img_seq = seq_img_join(x_samps, y_samps, im_dim=(28,28), transposed=True)
            seq_len = len(img_seq)
            samp_count = img_seq[0].shape[0]
            seq_samps = np.zeros((seq_len*samp_count, img_seq[0].shape[1]))
            idx = 0
            for s1 in range(samp_count):
                for s2 in range(seq_len):
                    if use_binary:
                        seq_samps[idx] = binarize_data(img_seq[s2][s1])
                    else:
                        seq_samps[idx] = img_seq[s2][s1]
                    idx += 1
            file_name = "{0:s}_samples_b{1:d}.png".format(res_tag, i)
            utils.visualize_samples(seq_samps, file_name, num_rows=samp_count)


if __name__=="__main__":
    #########################################################################
    # Train "binarized MNIST" generative models (open loopish LSTM quartet) #
    #########################################################################
    test_lstm_structpred(step_type='add', use_pol=True, use_binary=True)
    #test_lstm_structpred(step_type='add', use_pol=False, use_binary=True)
    #test_lstm_structpred(step_type='add', use_pol=False)








##############
# EYE BUFFER #
##############
