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
from TwoStageModel import TwoStageModel1, TwoStageModel2
from load_data import load_udm, load_binarized_mnist
import utils

#####################################
#####################################
## TEST MODEL THAT INFERS TOP-DOWN ##
#####################################
#####################################

def test_two_stage_model1():
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
    # Setup some parameters for the TwoStageModel #
    ###############################################
    x_dim = Xtr.shape[1]
    z_dim = 50
    h_dim = 100
    #h_det_dim = None
    h_det_dim = 50
    x_type = 'bernoulli'

    # some InfNet instances to build the TwoStageModel from
    xin_sym = T.matrix('xin_sym')
    xout_sym = T.matrix('xout_sym')

    ###############
    # p_h_given_z #
    ###############
    params = {}
    shared_config = [z_dim, 100, 100]
    shared_bn = [True, True]
    output_config = [h_dim, h_dim]
    params['shared_config'] = shared_config
    params['shared_bn'] = shared_bn
    params['output_config'] = output_config
    params['activation'] = tanh_actfun
    params['init_scale'] = 1.0
    params['lam_l2a'] = 0.0
    params['build_theano_funcs'] = False
    p_h_given_z = HydraNet(rng=rng, Xd=xin_sym, \
            params=params, shared_param_dicts=None)
    p_h_given_z.init_biases(0.0)
    ###############
    # p_x_given_h #
    ###############
    params = {}
    shared_config = [h_dim, 200, 200]
    shared_bn = [True, True]
    output_config = [x_dim, x_dim]
    params['shared_config'] = shared_config
    params['shared_bn'] = shared_bn
    params['output_config'] = output_config
    params['activation'] = tanh_actfun
    params['init_scale'] = 1.0
    params['lam_l2a'] = 0.0
    params['build_theano_funcs'] = False
    p_x_given_h = HydraNet(rng=rng, Xd=xin_sym, \
            params=params, shared_param_dicts=None)
    p_x_given_h.init_biases(0.0)
    ###############
    # q_z_given_x #
    ###############
    params = {}
    shared_config = [x_dim, 200, 200]
    shared_bn = [True, True]
    output_config = [z_dim, z_dim]
    params['shared_config'] = shared_config
    params['shared_bn'] = shared_bn
    params['output_config'] = output_config
    params['activation'] = tanh_actfun
    params['init_scale'] = 1.0
    params['lam_l2a'] = 0.0
    params['build_theano_funcs'] = False
    q_z_given_x = HydraNet(rng=rng, Xd=xin_sym, \
            params=params, shared_param_dicts=None)
    q_z_given_x.init_biases(0.0)
    #################
    # q_h_given_z_x #
    #################
    params = {}
    shared_config = [(h_dim + x_dim), 200, 200]
    shared_bn = [True, True]
    output_config = [h_dim, h_dim]
    params['shared_config'] = shared_config
    params['shared_bn'] = shared_bn
    params['output_config'] = output_config
    params['activation'] = tanh_actfun
    params['init_scale'] = 1.0
    params['lam_l2a'] = 0.0
    params['build_theano_funcs'] = False
    q_h_given_z_x = HydraNet(rng=rng, Xd=xin_sym, \
            params=params, shared_param_dicts=None)
    q_h_given_z_x.init_biases(0.0)

    ##############################################################
    # Define parameters for the TwoStageModel, and initialize it #
    ##############################################################
    print("Building the TwoStageModel...")
    tsm_params = {}
    tsm_params['x_type'] = x_type
    tsm_params['obs_transform'] = 'sigmoid'
    TSM = TwoStageModel1(rng=rng, x_in=xin_sym, x_out=xout_sym,
            x_dim=x_dim, z_dim=z_dim, h_dim=h_dim, h_det_dim=h_det_dim,
            q_z_given_x=q_z_given_x,
            q_h_given_z_x=q_h_given_z_x,
            p_h_given_z=p_h_given_z,
            p_x_given_h=p_x_given_h,
            params=tsm_params)

    ################################################################
    # Apply some updates, to check that they aren't totally broken #
    ################################################################
    log_name = "{}_RESULTS.txt".format("TSM1B_TEST")
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
        TSM.set_sgd_params(lr=scale*learn_rate, \
                           mom_1=(scale*momentum), mom_2=0.98)
        TSM.set_train_switch(1.0)
        TSM.set_lam_nll(lam_nll=1.0)
        TSM.set_lam_kld(lam_kld_q2p=1.0, lam_kld_p2q=0.0)
        TSM.set_lam_l2w(1e-5)
        # perform a minibatch update and record the cost for this batch
        result = TSM.train_joint(Xb, Xb, batch_reps)
        costs = [(costs[j] + result[j]) for j in range(len(result))]
        if ((i % 500) == 0):
            costs = [(v / 500.0) for v in costs]
            str1 = "-- batch {0:d} --".format(i)
            str2 = "    joint_cost: {0:.4f}".format(costs[0])
            str3 = "    nll_cost  : {0:.4f}".format(costs[1])
            str4 = "    kld_cost  : {0:.4f}".format(costs[2])
            str5 = "    reg_cost  : {0:.4f}".format(costs[3])
            str6 = "    nll       : {0:.4f}".format(np.mean(costs[4]))
            str7 = "    kld_z     : {0:.4f}".format(np.mean(costs[5]))
            str8 = "    kld_h     : {0:.4f}".format(np.mean(costs[6]))
            joint_str = "\n".join([str1, str2, str3, str4, str5, str6, str7, str8])
            print(joint_str)
            out_file.write(joint_str+"\n")
            out_file.flush()
            costs = [0.0 for v in costs]
        if (((i % 5000) == 0) or ((i < 10000) and ((i % 1000) == 0))):
            # draw some independent random samples from the model
            samp_count = 300
            model_samps = TSM.sample_from_prior(samp_count)
            file_name = "TSM1B_SAMPLES_b{0:d}.png".format(i)
            utils.visualize_samples(model_samps, file_name, num_rows=15)
            # compute free energy estimate for validation samples
            Xva = row_shuffle(Xva)
            fe_terms = TSM.compute_fe_terms(Xva[0:5000], Xva[0:5000], 20)
            fe_mean = np.mean(fe_terms[0]) + np.mean(fe_terms[1])
            out_str = "    nll_bound : {0:.4f}".format(fe_mean)
            print(out_str)
            out_file.write(out_str+"\n")
            out_file.flush()
    return

######################################
######################################
## TEST MODEL THAT INFERS BOTTOM-UP ##
######################################
######################################

def test_two_stage_model2():
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
    # Setup some parameters for the TwoStageModel #
    ###############################################
    x_dim = Xtr.shape[1]
    z_dim = 50
    h_dim = 100
    x_type = 'bernoulli'

    # some InfNet instances to build the TwoStageModel from
    xin_sym = T.matrix('xin_sym')
    xout_sym = T.matrix('xout_sym')

    ###############
    # p_h_given_z #
    ###############
    params = {}
    shared_config = [z_dim, 100, 100]
    shared_bn = [True, True]
    output_config = [h_dim, h_dim]
    params['shared_config'] = shared_config
    params['shared_bn'] = shared_bn
    params['output_config'] = output_config
    params['activation'] = tanh_actfun
    params['init_scale'] = 1.0
    params['lam_l2a'] = 0.0
    params['build_theano_funcs'] = False
    p_h_given_z = HydraNet(rng=rng, Xd=xin_sym,
            params=params, shared_param_dicts=None)
    p_h_given_z.init_biases(0.0)
    ###############
    # p_x_given_h #
    ###############
    params = {}
    shared_config = [h_dim, 200, 200]
    shared_bn = [True, True]
    output_config = [x_dim, x_dim]
    params['shared_config'] = shared_config
    params['shared_bn'] = shared_bn
    params['output_config'] = output_config
    params['activation'] = tanh_actfun
    params['init_scale'] = 1.0
    params['lam_l2a'] = 0.0
    params['build_theano_funcs'] = False
    p_x_given_h = HydraNet(rng=rng, Xd=xin_sym,
            params=params, shared_param_dicts=None)
    p_x_given_h.init_biases(0.0)
    ###############
    # q_h_given_x #
    ###############
    params = {}
    shared_config = [x_dim, 200, 200]
    shared_bn = [True, True]
    output_config = [h_dim, h_dim]
    params['shared_config'] = shared_config
    params['shared_bn'] = shared_bn
    params['output_config'] = output_config
    params['activation'] = tanh_actfun
    params['init_scale'] = 1.0
    params['lam_l2a'] = 0.0
    params['build_theano_funcs'] = False
    q_h_given_x = HydraNet(rng=rng, Xd=xin_sym,
            params=params, shared_param_dicts=None)
    q_h_given_x.init_biases(0.0)
    ###############
    # q_z_given_h #
    ###############
    params = {}
    shared_config = [h_dim, 100, 100]
    shared_bn = [True, True]
    output_config = [z_dim, z_dim]
    params['shared_config'] = shared_config
    params['shared_bn'] = shared_bn
    params['output_config'] = output_config
    params['activation'] = tanh_actfun
    params['init_scale'] = 1.0
    params['lam_l2a'] = 0.0
    params['build_theano_funcs'] = False
    q_z_given_h = HydraNet(rng=rng, Xd=xin_sym,
            params=params, shared_param_dicts=None)
    q_z_given_h.init_biases(0.0)

    ##############################################################
    # Define parameters for the TwoStageModel, and initialize it #
    ##############################################################
    print("Building the TwoStageModel...")
    tsm_params = {}
    tsm_params['x_type'] = x_type
    tsm_params['obs_transform'] = 'sigmoid'
    TSM = TwoStageModel2(rng=rng, x_in=xin_sym, x_out=xout_sym,
            x_dim=x_dim, z_dim=z_dim, h_dim=h_dim,
            q_h_given_x=q_h_given_x,
            q_z_given_h=q_z_given_h,
            p_h_given_z=p_h_given_z,
            p_x_given_h=p_x_given_h,
            params=tsm_params)

    ################################################################
    # Apply some updates, to check that they aren't totally broken #
    ################################################################
    log_name = "{}_RESULTS.txt".format("TSM2A_TEST")
    out_file = open(log_name, 'wb')
    costs = [0. for i in range(10)]
    learn_rate = 0.001
    momentum = 0.9
    batch_idx = np.arange(batch_size) + tr_samples
    for i in range(500000):
        scale = min(1.0, ((i+1) / 5000.0))
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
        TSM.set_sgd_params(lr=scale*learn_rate,
                           mom_1=(scale*momentum), mom_2=0.98)
        TSM.set_train_switch(1.0)
        TSM.set_lam_nll(lam_nll=1.0)
        TSM.set_lam_kld(lam_kld_q2p=1.0, lam_kld_p2q=0.0)
        TSM.set_lam_l2w(1e-5)
        # perform a minibatch update and record the cost for this batch
        result = TSM.train_joint(Xb, Xb, batch_reps)
        costs = [(costs[j] + result[j]) for j in range(len(result))]
        if ((i % 500) == 0):
            costs = [(v / 500.0) for v in costs]
            str1 = "-- batch {0:d} --".format(i)
            str2 = "    joint_cost: {0:.4f}".format(costs[0])
            str3 = "    nll_cost  : {0:.4f}".format(costs[1])
            str4 = "    kld_cost  : {0:.4f}".format(costs[2])
            str5 = "    reg_cost  : {0:.4f}".format(costs[3])
            str6 = "    nll       : {0:.4f}".format(np.mean(costs[4]))
            str7 = "    kld_z     : {0:.4f}".format(np.mean(costs[5]))
            str8 = "    kld_h     : {0:.4f}".format(np.mean(costs[6]))
            joint_str = "\n".join([str1, str2, str3, str4, str5, str6, str7, str8])
            print(joint_str)
            out_file.write(joint_str+"\n")
            out_file.flush()
            costs = [0.0 for v in costs]
        if (((i % 5000) == 0) or ((i < 10000) and ((i % 1000) == 0))):
            # draw some independent random samples from the model
            samp_count = 300
            model_samps = TSM.sample_from_prior(samp_count)
            file_name = "TSM2A_SAMPLES_b{0:d}.png".format(i)
            utils.visualize_samples(model_samps, file_name, num_rows=15)
            # compute free energy estimate for validation samples
            Xva = row_shuffle(Xva)
            fe_terms = TSM.compute_fe_terms(Xva[0:5000], Xva[0:5000], 20)
            fe_mean = np.mean(fe_terms[0]) + np.mean(fe_terms[1])
            out_str = "    nll_bound : {0:.4f}".format(fe_mean)
            print(out_str)
            out_file.write(out_str+"\n")
            out_file.flush()
    return


if __name__=="__main__":
    test_two_stage_model1()
    #test_two_stage_model2()
