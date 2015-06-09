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
from HelperFuncs import construct_masked_data, shift_and_scale_into_01, \
                        row_shuffle, to_fX
from DKCode import get_adam_updates, get_adadelta_updates
from load_data import load_udm, load_tfd, load_mnist, load_svhn_gray

#########
# MNIST #
#########

def test_imocld_mnist(step_type='add', attention=False):
    ##########################
    # Get some training data #
    ##########################
    rng = np.random.RandomState(1234)
    dataset = 'data/mnist.pkl.gz'
    datasets = load_udm(dataset, as_shared=False, zero_mean=False)
    Xtr = datasets[0][0]
    Xva = datasets[1][0]
    Xtr = to_fX(shift_and_scale_into_01(Xtr))
    Xva = to_fX(shift_and_scale_into_01(Xva))
    tr_samples = Xtr.shape[0]
    va_samples = Xva.shape[0]
    batch_size = 250

    ############################################################
    # Setup some parameters for the Iterative Refinement Model #
    ############################################################
    x_dim = Xtr.shape[1]
    write_dim = 300
    enc_dim = 300
    dec_dim = 300
    mix_dim = 20
    z_dim = 100
    n_iter = 16
    
    rnninits = {
        'weights_init': IsotropicGaussian(0.01),
        'biases_init': Constant(0.),
    }
    inits = {
        'weights_init': IsotropicGaussian(0.01),
        'biases_init': Constant(0.),
    }

    att_tag = "NA" # attention not implemented yet

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
                      [mix_dim, 250, (2*enc_dim + 2*dec_dim + 2*enc_dim + mix_dim)], \
                      name="mix_dec_mlp", **inits)
    # mlps for processing inputs to LSTMs
    var_mlp_in = MLP([Identity()], [(read_dim + dec_dim + mix_dim), 4*enc_dim], \
                     name="var_mlp_in", **inits)
    enc_mlp_in = MLP([Identity()], [(read_dim + dec_dim + mix_dim), 4*enc_dim], \
                     name="enc_mlp_in", **inits)
    dec_mlp_in = MLP([Identity()], [                         z_dim, 4*dec_dim], \
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

    draw = IMoCLDrawModels(
                n_iter,
                step_type=step_type, # step_type can be 'add' or 'jump'
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
    draw.initialize()
    # build the cost gradients, training function, samplers, etc.
    draw.build_model_funcs()

    # sample several interchangeable versions of the model
    conditions = [{'occ_dim': 0, 'drop_prob': 0.8}, \
                  {'occ_dim': 16, 'drop_prob': 0.0}]
    for cond_dict in conditions:
        occ_dim = cond_dict['occ_dim']
        drop_prob = cond_dict['drop_prob']
        dp_int = int(100.0 * drop_prob)

        draw.load_model_params(f_name="TBCLM_IMP_MNIST_PARAMS_OD{}_DP{}_{}_{}.pkl".format(occ_dim, dp_int, step_type, att_tag))

        # draw some independent samples from the model
        Xva = row_shuffle(Xva)
        Xb = to_fX(Xva[:128])
        _, Xb, Mb = construct_masked_data(Xb, drop_prob=drop_prob, \
                                occ_dim=occ_dim, data_mean=None)
        Xb = np.repeat(Xb, 2, axis=0)
        Mb = np.repeat(Mb, 2, axis=0)
        samples = draw.do_sample(Xb, Mb)

        # save the samples to a pkl file, in their numpy array form
        sample_pkl_name = "IMP-MNIST-OD{0:d}-DP{1:d}-{2:s}.pkl".format(occ_dim, dp_int, step_type)
        f_handle = file(sample_pkl_name, 'wb')
        cPickle.dump(samples, f_handle, protocol=-1)
        f_handle.close()
        print("Saved some samples in: {}".format(sample_pkl_name))
    return


#######
# TFD #
#######

def test_imocld_tfd(step_type='add', attention=False):
    ##########################
    # Get some training data #
    ##########################
    rng = np.random.RandomState(1234)
    data_file = 'data/tfd_data_48x48.pkl'
    dataset = load_tfd(tfd_pkl_name=data_file, which_set='unlabeled', fold='all')
    Xtr_unlabeled = dataset[0]
    dataset = load_tfd(tfd_pkl_name=data_file, which_set='train', fold='all')
    Xtr_train = dataset[0]
    Xtr = np.vstack([Xtr_unlabeled, Xtr_train])
    dataset = load_tfd(tfd_pkl_name=data_file, which_set='valid', fold='all')
    Xva = dataset[0]
    Xtr = to_fX(shift_and_scale_into_01(Xtr))
    Xva = to_fX(shift_and_scale_into_01(Xva))
    tr_samples = Xtr.shape[0]
    va_samples = Xva.shape[0]
    batch_size = 250
    batch_reps = 1
    all_pix_mean = np.mean(np.mean(Xtr, axis=1))
    data_mean = to_fX( all_pix_mean * np.ones((Xtr.shape[1],)) )

    ############################################################
    # Setup some parameters for the Iterative Refinement Model #
    ############################################################
    x_dim = Xtr.shape[1]
    write_dim = 600
    enc_dim = 600
    dec_dim = 600
    mix_dim = 20
    z_dim = 200
    n_iter = 16
    
    rnninits = {
        'weights_init': IsotropicGaussian(0.01),
        'biases_init': Constant(0.),
    }
    inits = {
        'weights_init': IsotropicGaussian(0.01),
        'biases_init': Constant(0.),
    }

    att_tag = "NA" # attention not implemented yet

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

    draw = IMoCLDrawModels(
                n_iter,
                step_type=step_type, # step_type can be 'add' or 'jump'
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
    draw.initialize()

    # build the cost gradients, training function, samplers, etc.
    draw.build_model_funcs()

    

    # sample several interchangeable versions of the model
    conditions = [{'occ_dim': 0, 'drop_prob': 0.8}, \
                  {'occ_dim': 25, 'drop_prob': 0.0}]
    for cond_dict in conditions:
        occ_dim = cond_dict['occ_dim']
        drop_prob = cond_dict['drop_prob']
        dp_int = int(100.0 * drop_prob)

        draw.load_model_params(f_name="TBCLM_IMP_TFD_PARAMS_OD{}_DP{}_{}_{}.pkl".format(occ_dim, dp_int, step_type, att_tag))

        # draw some independent samples from the model
        Xva = row_shuffle(Xva)
        Xb = to_fX(Xva[:128])
        _, Xb, Mb = construct_masked_data(Xb, drop_prob=drop_prob, \
                                occ_dim=occ_dim, data_mean=None)
        Xb = np.repeat(Xb, 2, axis=0)
        Mb = np.repeat(Mb, 2, axis=0)
        samples = draw.do_sample(Xb, Mb)

        # save the samples to a pkl file, in their numpy array form
        sample_pkl_name = "IMP-TFD-OD{0:d}-DP{1:d}-{2:s}.pkl".format(occ_dim, dp_int, step_type)
        f_handle = file(sample_pkl_name, 'wb')
        cPickle.dump(samples, f_handle, protocol=-1)
        f_handle.close()
        print("Saved some samples in: {}".format(sample_pkl_name))
    return

########
# SVHN #
########

def test_imocld_svhn(step_type='add', attention=False):
    ##########################
    # Get some training data #
    ##########################
    rng = np.random.RandomState(1234)
    tr_file = 'data/svhn_train_gray.pkl'
    te_file = 'data/svhn_test_gray.pkl'
    ex_file = 'data/svhn_extra_gray.pkl'
    data = load_svhn_gray(tr_file, te_file, ex_file=ex_file, ex_count=200000)
    Xtr = to_fX( shift_and_scale_into_01(np.vstack([data['Xtr'], data['Xex']])) )
    Xva = to_fX( shift_and_scale_into_01(data['Xte']) )
    tr_samples = Xtr.shape[0]
    va_samples = Xva.shape[0]
    batch_size = 250
    all_pix_mean = np.mean(np.mean(Xtr, axis=1))
    data_mean = to_fX( all_pix_mean * np.ones((Xtr.shape[1],)) )

    ############################################################
    # Setup some parameters for the Iterative Refinement Model #
    ############################################################
    x_dim = Xtr.shape[1]
    write_dim = 600
    enc_dim = 600
    dec_dim = 600
    mix_dim = 20
    z_dim = 200
    n_iter = 16
    
    rnninits = {
        'weights_init': IsotropicGaussian(0.01),
        'biases_init': Constant(0.),
    }
    inits = {
        'weights_init': IsotropicGaussian(0.01),
        'biases_init': Constant(0.),
    }

    att_tag = "NA" # attention not implemented yet

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

    draw = IMoCLDrawModels(
                n_iter,
                step_type=step_type, # step_type can be 'add' or 'jump'
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
    draw.initialize()

    # build the cost gradients, training function, samplers, etc.
    draw.build_model_funcs()

    # sample several interchangeable versions of the model
    conditions = [{'occ_dim': 0, 'drop_prob': 0.8}, \
                  {'occ_dim': 17, 'drop_prob': 0.0}]
    for cond_dict in conditions:
        occ_dim = cond_dict['occ_dim']
        drop_prob = cond_dict['drop_prob']
        dp_int = int(100.0 * drop_prob)

        draw.load_model_params(f_name="TBCLM_IMP_SVHN_PARAMS_OD{}_DP{}_{}_{}.pkl".format(occ_dim, dp_int, step_type, att_tag))

        # draw some independent samples from the model
        Xva = row_shuffle(Xva)
        Xb = to_fX(Xva[:128])
        _, Xb, Mb = construct_masked_data(Xb, drop_prob=drop_prob, \
                                occ_dim=occ_dim, data_mean=None)
        Xb = np.repeat(Xb, 2, axis=0)
        Mb = np.repeat(Mb, 2, axis=0)
        samples = draw.do_sample(Xb, Mb)

        # save the samples to a pkl file, in their numpy array form
        sample_pkl_name = "IMP-SVHN-OD{0:d}-DP{1:d}-{2:s}.pkl".format(occ_dim, dp_int, step_type)
        f_handle = file(sample_pkl_name, 'wb')
        cPickle.dump(samples, f_handle, protocol=-1)
        f_handle.close()
        print("Saved some samples in: {}".format(sample_pkl_name))
    return


if __name__=="__main__":
    #test_imocld_mnist(step_type='add')
    #test_imocld_mnist(step_type='jump')
    test_imocld_tfd(step_type='add')
    test_imocld_svhn(step_type='add')
