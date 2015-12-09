##################################################################
# Code for testing the variational Multi-Stage Generative Model. #
##################################################################

# basic python
import numpy as np
import numpy.random as npr

# theano business
import theano
import theano.tensor as T

# phil's sweetness
from LogPDFs import log_prob_bernoulli, log_prob_gaussian2, gaussian_kld
from NetLayers import relu_actfun, softplus_actfun, tanh_actfun
from HelperFuncs import apply_mask, binarize_data, row_shuffle, to_fX
from HydraNet import HydraNet
from OneStageModel import OneStageModel
from load_data import load_udm, load_binarized_mnist
import utils

#####################################
#####################################
## TEST MODEL THAT INFERS TOP-DOWN ##
#####################################
#####################################

def test_one_stage_model():
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
    batch_size = 128
    batch_reps = 1

    ###############################################
    # Setup some parameters for the OneStageModel #
    ###############################################
    x_dim = Xtr.shape[1]
    z_dim = 64
    x_type = 'bernoulli'
    xin_sym = T.matrix('xin_sym')

    ###############
    # p_x_given_z #
    ###############
    params = {}
    shared_config = \
    [ {'layer_type': 'fc',
       'in_chans': z_dim,
       'out_chans': 256,
       'activation': relu_actfun,
       'apply_bn': True}, \
      {'layer_type': 'fc',
       'in_chans': 256,
       'out_chans': 7*7*128,
       'activation': relu_actfun,
       'apply_bn': True,
       'shape_func_out': lambda x: T.reshape(x, (-1, 128, 7, 7))}, \
      {'layer_type': 'conv',
       'in_chans': 128, # in shape:  (batch, 128, 7, 7)
       'out_chans': 64, # out shape: (batch, 64, 14, 14)
       'activation': relu_actfun,
       'filt_dim': 5,
       'conv_stride': 'half',
       'apply_bn': True} ]
    output_config = \
    [ {'layer_type': 'conv',
       'in_chans': 64, # in shape:  (batch, 64, 14, 14)
       'out_chans': 1, # out shape: (batch, 1, 28, 28)
       'activation': relu_actfun,
       'filt_dim': 5,
       'conv_stride': 'half',
       'apply_bn': False,
       'shape_func_out': lambda x: T.flatten(x, 2)}, \
      {'layer_type': 'conv',
       'in_chans': 64,
       'out_chans': 1,
       'activation': relu_actfun,
       'filt_dim': 5,
       'conv_stride': 'half',
       'apply_bn': False,
       'shape_func_out': lambda x: T.flatten(x, 2)} ]
    params['shared_config'] = shared_config
    params['output_config'] = output_config
    params['init_scale'] = 1.0
    params['build_theano_funcs'] = False
    p_x_given_z = HydraNet(rng=rng, Xd=xin_sym, \
            params=params, shared_param_dicts=None)
    p_x_given_z.init_biases(0.0)
    ###############
    # q_z_given_x #
    ###############
    params = {}
    shared_config = \
    [ {'layer_type': 'conv',
       'in_chans': 1,   # in shape:  (batch, 784)
       'out_chans': 64, # out shape: (batch, 64, 14, 14)
       'activation': relu_actfun,
       'filt_dim': 5,
       'conv_stride': 'double',
       'apply_bn': True,
       'shape_func_in': lambda x: T.reshape(x, (-1, 1, 28, 28))}, \
      {'layer_type': 'conv',
       'in_chans': 64,   # in shape:  (batch, 64, 14, 14)
       'out_chans': 128, # out shape: (batch, 128, 7, 7)
       'activation': relu_actfun,
       'filt_dim': 5,
       'conv_stride': 'double',
       'apply_bn': True,
       'shape_func_out': lambda x: T.flatten(x, 2)}, \
      {'layer_type': 'fc',
       'in_chans': 128*7*7,
       'out_chans': 256,
       'activation': relu_actfun,
       'apply_bn': True} ]
    output_config = \
    [ {'layer_type': 'fc',
       'in_chans': 256,
       'out_chans': z_dim,
       'activation': relu_actfun,
       'apply_bn': False}, \
      {'layer_type': 'fc',
       'in_chans': 256,
       'out_chans': z_dim,
       'activation': relu_actfun,
       'apply_bn': False} ]
    params['shared_config'] = shared_config
    params['output_config'] = output_config
    params['init_scale'] = 1.0
    params['build_theano_funcs'] = False
    q_z_given_x = HydraNet(rng=rng, Xd=xin_sym, \
            params=params, shared_param_dicts=None)
    q_z_given_x.init_biases(0.0)


    ##############################################################
    # Define parameters for the TwoStageModel, and initialize it #
    ##############################################################
    print("Building the OneStageModel...")
    osm_params = {}
    osm_params['x_type'] = x_type
    osm_params['obs_transform'] = 'sigmoid'
    OSM = OneStageModel(rng=rng, x_in=xin_sym,
            x_dim=x_dim, z_dim=z_dim,
            p_x_given_z=p_x_given_z,
            q_z_given_x=q_z_given_x,
            params=osm_params)

    ################################################################
    # Apply some updates, to check that they aren't totally broken #
    ################################################################
    log_name = "{}_RESULTS.txt".format("OSM_TEST")
    out_file = open(log_name, 'wb')
    costs = [0. for i in range(10)]
    learn_rate = 0.0005
    momentum = 0.9
    batch_idx = np.arange(batch_size) + tr_samples
    for i in range(500000):
        scale = min(0.5, ((i+1) / 5000.0))
        if (((i + 1) % 10000) == 0):
            learn_rate = learn_rate * 0.95
        # get the indices of training samples for this batch update
        batch_idx += batch_size
        if (np.max(batch_idx) >= tr_samples):
            # we finished an "epoch", so we rejumble the training set
            Xtr = row_shuffle(Xtr)
            batch_idx = np.arange(batch_size)
        Xb = to_fX( Xtr.take(batch_idx, axis=0) )
        #Xb = binarize_data(Xtr.take(batch_idx, axis=0))
        # set sgd and objective function hyperparams for this update
        OSM.set_sgd_params(lr=scale*learn_rate, \
                           mom_1=(scale*momentum), mom_2=0.98)
        OSM.set_lam_nll(lam_nll=1.0)
        OSM.set_lam_kld(lam_kld=1.0)
        OSM.set_lam_l2w(1e-5)
        # perform a minibatch update and record the cost for this batch
        result = OSM.train_joint(Xb, batch_reps)
        costs = [(costs[j] + result[j]) for j in range(len(result))]
        if ((i % 500) == 0):
            costs = [(v / 500.0) for v in costs]
            str1 = "-- batch {0:d} --".format(i)
            str2 = "    joint_cost: {0:.4f}".format(costs[0])
            str3 = "    nll_cost  : {0:.4f}".format(costs[1])
            str4 = "    kld_cost  : {0:.4f}".format(costs[2])
            str5 = "    reg_cost  : {0:.4f}".format(costs[3])
            joint_str = "\n".join([str1, str2, str3, str4, str5])
            print(joint_str)
            out_file.write(joint_str+"\n")
            out_file.flush()
            costs = [0.0 for v in costs]
        if (((i % 5000) == 0) or ((i < 10000) and ((i % 1000) == 0))):
            # draw some independent random samples from the model
            samp_count = 300
            model_samps = OSM.sample_from_prior(samp_count)
            file_name = "OSM_SAMPLES_b{0:d}.png".format(i)
            utils.visualize_samples(model_samps, file_name, num_rows=15)
            # compute free energy estimate for validation samples
            Xva = row_shuffle(Xva)
            fe_terms = OSM.compute_fe_terms(Xva[0:5000], 20)
            fe_mean = np.mean(fe_terms[0]) + np.mean(fe_terms[1])
            out_str = "    nll_bound : {0:.4f}".format(fe_mean)
            print(out_str)
            out_file.write(out_str+"\n")
            out_file.flush()
    return



if __name__=="__main__":
    test_one_stage_model()
