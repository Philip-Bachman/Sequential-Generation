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
from HelperFuncs import row_shuffle, to_fX

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


#############################
#############################
## TEST BASIC RLDRAW MODEL ##
#############################
#############################

def test_rldraw_classic(step_type='add', use_pol=True):
    ###########################################
    # Make a tag for identifying result files #
    ###########################################
    pol_tag = "yp" if use_pol else "np"
    res_tag = "TRLD_{}_{}".format(step_type, pol_tag)

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
    write_dim = 250
    pol_dim = 250
    enc_dim = 250
    dec_dim = 250
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

    # setup reader/writer models
    read_dim = 2*x_dim
    reader_mlp = Reader(x_dim=x_dim, dec_dim=dec_dim, **inits)
    writer_mlp = MLP([None, None], [dec_dim, write_dim, x_dim],
                     name="writer_mlp", **inits)

    # setup submodels for processing LSTM inputs
    pol_mlp_in = MLP([Identity()], [dec_dim, 4*pol_dim],
                     name="pol_mlp_in", **inits)
    enc_mlp_in = MLP([Identity()], [(x_dim + dec_dim), 4*enc_dim],
                     name="enc_mlp_in", **inits)
    dec_mlp_in = MLP([Identity()], [z_dim, 4*dec_dim],
                     name="dec_mlp_in", **inits)
    # setup submodels for turning LSTM states into conditionals over z
    pol_mlp_out = CondNet([], [pol_dim, z_dim], name="pol_mlp_out", **inits)
    enc_mlp_out = CondNet([], [enc_dim, z_dim], name="enc_mlp_out", **inits)
    dec_mlp_out = CondNet([], [dec_dim, z_dim], name="dec_mlp_out", **inits)
    # setup the LSTMs for primary policy, guide policy, and shared dynamics
    pol_rnn = BiasedLSTM(dim=pol_dim, ig_bias=2.0, fg_bias=2.0, \
                         name="pol_rnn", **rnninits)
    enc_rnn = BiasedLSTM(dim=enc_dim, ig_bias=2.0, fg_bias=2.0, \
                         name="enc_rnn", **rnninits)
    dec_rnn = BiasedLSTM(dim=dec_dim, ig_bias=2.0, fg_bias=2.0, \
                         name="dec_rnn", **rnninits)

    draw = RLDrawModel(
                n_iter,
                step_type=step_type, # step_type can be 'add' or 'jump'
                use_pol=use_pol,
                reader_mlp=reader_mlp,
                writer_mlp=writer_mlp,
                pol_mlp_in=pol_mlp_in,
                pol_mlp_out=pol_mlp_out,
                pol_rnn=pol_rnn,
                enc_mlp_in=enc_mlp_in,
                enc_mlp_out=enc_mlp_out,
                enc_rnn=enc_rnn,
                dec_mlp_in=dec_mlp_in,
                dec_mlp_out=dec_mlp_out,
                dec_rnn=dec_rnn)
    draw.initialize()

    compile_start_time = time.time()

    # build the cost gradients, training function, samplers, etc.
    draw.build_sampling_funcs()
    print("Testing model sampler...")
    # draw some independent samples from the model
    samples = draw.sample_model(Xtr[:65,:], sample_source='p')
    n_iter, N, D = samples.shape
    samples = samples.reshape( (n_iter, N, 28, 28) )
    for j in xrange(n_iter):
        img = img_grid(samples[j,:,:,:])
        img.save("%s_samples_%03d.png" % (res_tag, j))

    draw.build_model_funcs()

    compile_end_time = time.time()
    compile_minutes = (compile_end_time - compile_start_time) / 60.0
    print("THEANO COMPILE TIME (MIN): {}".format(compile_minutes))

    ################################################################
    # Apply some updates, to check that they aren't totally broken #
    ################################################################
    print("Beginning to train the model...")
    out_file = open("{}_results.txt".format(res_tag), 'wb')
    costs = [0. for i in range(10)]
    learn_rate = 0.00015
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
            Xtr = row_shuffle(Xtr)
            batch_idx = np.arange(batch_size)
        # set sgd and objective function hyperparams for this update
        draw.set_sgd_params(lr=scale*learn_rate, mom_1=scale*momentum, mom_2=0.98)
        # perform a minibatch update and record the cost for this batch
        Xb = to_fX(Xtr.take(batch_idx, axis=0))
        result = draw.train_joint(Xb)
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
            draw.save_model_params("{}_params.pkl".format(res_tag))
            # compute a small-sample estimate of NLL bound on validation set
            Xva = row_shuffle(Xva)
            Xb = to_fX(Xva[:5000])
            va_costs = draw.compute_nll_bound(Xb)
            str1 = "    va_nll_bound : {}".format(va_costs[1])
            str2 = "    va_nll_term  : {}".format(va_costs[2])
            str3 = "    va_kld_q2p   : {}".format(va_costs[3])
            joint_str = "\n".join([str1, str2, str3])
            print(joint_str)
            out_file.write(joint_str+"\n")
            out_file.flush()
            # draw some independent samples from the model
            samples = draw.sample_model(Xb[:256,:], sample_source='p')
            n_iter, N, D = samples.shape
            samples = samples.reshape( (n_iter, N, 28, 28) )
            for j in xrange(n_iter):
                img = img_grid(samples[j,:,:,:])
                img.save("%s_samples_%03d.png" % (res_tag, j))


if __name__=="__main__":
    #########################################################################
    # Train "binarized MNIST" generative models (open loopish LSTM quartet) #
    #########################################################################
    #test_rldraw_classic(step_type='add', use_pol=True)
    test_rldraw_classic(step_type='add', use_pol=False)








##############
# EYE BUFFER #
##############
