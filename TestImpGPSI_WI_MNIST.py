##################################################################
# Code for testing the variational Multi-Stage Generative Model. #
##################################################################

# basic python
import numpy as np
import numpy.random as npr
import cPickle

# theano business
import theano
import theano.tensor as T

# phil's sweetness
import utils
from NetLayers import relu_actfun, softplus_actfun, tanh_actfun
from InfNet import InfNet
from HydraNet import HydraNet
from GPSImputerWI import GPSImputerWI, load_gpsimputer_from_file
from load_data import load_udm, load_tfd, load_svhn_gray, load_binarized_mnist
from HelperFuncs import construct_masked_data, shift_and_scale_into_01, \
                        row_shuffle, to_fX

RESULT_PATH = "IMP_MNIST_GPSI_WI/"

###############################
###############################
## TEST GPS IMPUTER ON MNIST ##
###############################
###############################

def test_mnist(step_type='add',
               imp_steps=6,
               occ_dim=15,
               drop_prob=0.0):
    #########################################
    # Format the result tag more thoroughly #
    #########################################
    dp_int = int(100.0 * drop_prob)
    result_tag = "{}GPSI_OD{}_DP{}_IS{}_{}_NA".format(RESULT_PATH, occ_dim, dp_int, imp_steps, step_type)

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

    ##########################
    # Get some training data #
    ##########################
    # rng = np.random.RandomState(1234)
    # dataset = 'data/mnist.pkl.gz'
    # datasets = load_udm(dataset, as_shared=False, zero_mean=False)
    # Xtr = datasets[0][0]
    # Xva = datasets[1][0]
    # Xte = datasets[2][0]
    # # Merge validation set and training set, and test on test set.
    # #Xtr = np.concatenate((Xtr, Xva), axis=0)
    # #Xva = Xte
    # Xtr = to_fX(shift_and_scale_into_01(Xtr))
    # Xva = to_fX(shift_and_scale_into_01(Xva))
    # tr_samples = Xtr.shape[0]
    # va_samples = Xva.shape[0]
    batch_size = 200
    batch_reps = 1
    all_pix_mean = np.mean(np.mean(Xtr, axis=1))
    data_mean = to_fX( all_pix_mean * np.ones((Xtr.shape[1],)) )

    ############################################################
    # Setup some parameters for the Iterative Refinement Model #
    ############################################################
    x_dim = Xtr.shape[1]
    s_dim = x_dim
    h_dim = 50
    z_dim = 100
    init_scale = 0.6

    x_in_sym = T.matrix('x_in_sym')
    x_out_sym = T.matrix('x_out_sym')
    x_mask_sym = T.matrix('x_mask_sym')

    ###############
    # p_h_given_x #
    ###############
    params = {}
    shared_config = [x_dim, 250]
    top_config = [shared_config[-1], h_dim]
    params['shared_config'] = shared_config
    params['mu_config'] = top_config
    params['sigma_config'] = top_config
    params['activation'] = tanh_actfun #relu_actfun
    params['init_scale'] = init_scale
    params['vis_drop'] = 0.0
    params['hid_drop'] = 0.0
    params['bias_noise'] = 0.0
    params['input_noise'] = 0.0
    params['build_theano_funcs'] = False
    p_h_given_x = InfNet(rng=rng, Xd=x_in_sym, \
            params=params, shared_param_dicts=None)
    p_h_given_x.init_biases(0.0)
    ################
    # p_s0_given_h #
    ################
    params = {}
    shared_config = [h_dim, 250]
    output_config = [s_dim, s_dim, s_dim]
    params['shared_config'] = shared_config
    params['output_config'] = output_config
    params['activation'] = tanh_actfun #relu_actfun
    params['init_scale'] = init_scale
    params['vis_drop'] = 0.0
    params['hid_drop'] = 0.0
    params['bias_noise'] = 0.0
    params['input_noise'] = 0.0
    params['build_theano_funcs'] = False
    p_s0_given_h = HydraNet(rng=rng, Xd=x_in_sym, \
            params=params, shared_param_dicts=None)
    p_s0_given_h.init_biases(0.0)
    #################
    # p_zi_given_xi #
    #################
    params = {}
    shared_config = [(x_dim + x_dim), 500, 500]
    top_config = [shared_config[-1], z_dim]
    params['shared_config'] = shared_config
    params['mu_config'] = top_config
    params['sigma_config'] = top_config
    params['activation'] = tanh_actfun #relu_actfun
    params['init_scale'] = init_scale
    params['vis_drop'] = 0.0
    params['hid_drop'] = 0.0
    params['bias_noise'] = 0.0
    params['input_noise'] = 0.0
    params['build_theano_funcs'] = False
    p_zi_given_xi = InfNet(rng=rng, Xd=x_in_sym, \
            params=params, shared_param_dicts=None)
    p_zi_given_xi.init_biases(0.0)
    ###################
    # p_sip1_given_zi #
    ###################
    params = {}
    shared_config = [z_dim, 500, 500]
    output_config = [s_dim, s_dim, s_dim]
    params['shared_config'] = shared_config
    params['output_config'] = output_config
    params['activation'] = tanh_actfun #relu_actfun
    params['init_scale'] = init_scale
    params['vis_drop'] = 0.0
    params['hid_drop'] = 0.0
    params['bias_noise'] = 0.0
    params['input_noise'] = 0.0
    params['build_theano_funcs'] = False
    p_sip1_given_zi = HydraNet(rng=rng, Xd=x_in_sym, \
            params=params, shared_param_dicts=None)
    p_sip1_given_zi.init_biases(0.0)
    ################
    # p_x_given_si #
    ################
    params = {}
    shared_config = [s_dim]
    output_config = [x_dim, x_dim]
    params['shared_config'] = shared_config
    params['output_config'] = output_config
    params['activation'] = tanh_actfun #relu_actfun
    params['init_scale'] = init_scale
    params['vis_drop'] = 0.0
    params['hid_drop'] = 0.0
    params['bias_noise'] = 0.0
    params['input_noise'] = 0.0
    params['build_theano_funcs'] = False
    p_x_given_si = HydraNet(rng=rng, Xd=x_in_sym, \
            params=params, shared_param_dicts=None)
    p_x_given_si.init_biases(0.0)
    ###############
    # q_h_given_x #
    ###############
    params = {}
    shared_config = [x_dim, 250]
    top_config = [shared_config[-1], h_dim]
    params['shared_config'] = shared_config
    params['mu_config'] = top_config
    params['sigma_config'] = top_config
    params['activation'] = tanh_actfun #relu_actfun
    params['init_scale'] = init_scale
    params['vis_drop'] = 0.0
    params['hid_drop'] = 0.0
    params['bias_noise'] = 0.0
    params['input_noise'] = 0.0
    params['build_theano_funcs'] = False
    q_h_given_x = InfNet(rng=rng, Xd=x_in_sym, \
            params=params, shared_param_dicts=None)
    q_h_given_x.init_biases(0.0)
    #################
    # q_zi_given_xi #
    #################
    params = {}
    shared_config = [(x_dim + x_dim), 500, 500]
    top_config = [shared_config[-1], z_dim]
    params['shared_config'] = shared_config
    params['mu_config'] = top_config
    params['sigma_config'] = top_config
    params['activation'] = tanh_actfun #relu_actfun
    params['init_scale'] = init_scale
    params['vis_drop'] = 0.0
    params['hid_drop'] = 0.0
    params['bias_noise'] = 0.0
    params['input_noise'] = 0.0
    params['build_theano_funcs'] = False
    q_zi_given_xi = InfNet(rng=rng, Xd=x_in_sym, \
            params=params, shared_param_dicts=None)
    q_zi_given_xi.init_biases(0.0)

    ###########################################################
    # Define parameters for the GPSImputer, and initialize it #
    ###########################################################
    print("Building the GPSImputer...")
    gpsi_params = {}
    gpsi_params['x_dim'] = x_dim
    gpsi_params['h_dim'] = h_dim
    gpsi_params['z_dim'] = z_dim
    gpsi_params['s_dim'] = s_dim
    # switch between direct construction and construction via p_x_given_si
    gpsi_params['use_p_x_given_si'] = False
    gpsi_params['imp_steps'] = imp_steps
    gpsi_params['step_type'] = step_type
    gpsi_params['x_type'] = 'bernoulli'
    gpsi_params['obs_transform'] = 'sigmoid'
    GPSI = GPSImputerWI(rng=rng,
            x_in=x_in_sym, x_out=x_out_sym, x_mask=x_mask_sym, \
            p_h_given_x=p_h_given_x, \
            p_s0_given_h=p_s0_given_h, \
            p_zi_given_xi=p_zi_given_xi, \
            p_sip1_given_zi=p_sip1_given_zi, \
            p_x_given_si=p_x_given_si, \
            q_h_given_x=q_h_given_x, \
            q_zi_given_xi=q_zi_given_xi, \
            params=gpsi_params, \
            shared_param_dicts=None)

    ################################################################
    # Apply some updates, to check that they aren't totally broken #
    ################################################################
    log_name = "{}_RESULTS.txt".format(result_tag)
    out_file = open(log_name, 'wb')
    costs = [0. for i in range(10)]
    learn_rate = 0.0002
    momentum = 0.5
    batch_idx = np.arange(batch_size) + tr_samples
    for i in range(250000):
        scale = min(1.0, ((i+1) / 5000.0))
        lam_scale = 1.0 - min(1.0, ((i+1) / 100000.0)) # decays from 1.0->0.0
        if (((i + 1) % 15000) == 0):
            learn_rate = learn_rate * 0.93
        if (i > 10000):
            momentum = 0.90
        else:
            momentum = 0.75
        # get the indices of training samples for this batch update
        batch_idx += batch_size
        if (np.max(batch_idx) >= tr_samples):
            # we finished an "epoch", so we rejumble the training set
            Xtr = row_shuffle(Xtr)
            batch_idx = np.arange(batch_size)
        # set sgd and objective function hyperparams for this update
        GPSI.set_sgd_params(lr=scale*learn_rate, \
                            mom_1=scale*momentum, mom_2=0.98)
        GPSI.set_train_switch(1.0)
        GPSI.set_lam_nll(lam_nll=1.0)
        GPSI.set_lam_kld(lam_kld_p=0.05, lam_kld_q=0.95, \
                         lam_kld_g=(0.1 * lam_scale), lam_kld_s=(0.1 * lam_scale))
        GPSI.set_lam_l2w(1e-5)
        # perform a minibatch update and record the cost for this batch
        xb = to_fX( Xtr.take(batch_idx, axis=0) )
        xi, xo, xm = construct_masked_data(xb, drop_prob=drop_prob, \
                                        occ_dim=occ_dim, data_mean=data_mean)
        result = GPSI.train_joint(xi, xo, xm, batch_reps)
        # do diagnostics and general training tracking
        costs = [(costs[j] + result[j]) for j in range(len(result)-1)]
        if ((i % 250) == 0):
            costs = [(v / 250.0) for v in costs]
            str1 = "-- batch {0:d} --".format(i)
            str2 = "    joint_cost: {0:.4f}".format(costs[0])
            str3 = "    nll_bound : {0:.4f}".format(costs[1])
            str4 = "    nll_cost  : {0:.4f}".format(costs[2])
            str5 = "    kld_cost  : {0:.4f}".format(costs[3])
            str6 = "    reg_cost  : {0:.4f}".format(costs[4])
            joint_str = "\n".join([str1, str2, str3, str4, str5, str6])
            print(joint_str)
            out_file.write(joint_str+"\n")
            out_file.flush()
            costs = [0.0 for v in costs]
        if ((i % 1000) == 0):
            Xva = row_shuffle(Xva)
            # record an estimate of performance on the test set
            xi, xo, xm = construct_masked_data(Xva[0:5000], drop_prob=drop_prob, \
                                               occ_dim=occ_dim, data_mean=data_mean)
            nll, kld = GPSI.compute_fe_terms(xi, xo, xm, sample_count=10)
            vfe = np.mean(nll) + np.mean(kld)
            str1 = "    va_nll_bound : {}".format(vfe)
            str2 = "    va_nll_term  : {}".format(np.mean(nll))
            str3 = "    va_kld_q2p   : {}".format(np.mean(kld))
            joint_str = "\n".join([str1, str2, str3])
            print(joint_str)
            out_file.write(joint_str+"\n")
            out_file.flush()
        if ((i % 2000) == 0):
            GPSI.save_to_file("{}_PARAMS.pkl".format(result_tag))
            # Get some validation samples for evaluating model performance
            xb = to_fX( Xva[0:100] )
            xi, xo, xm = construct_masked_data(xb, drop_prob=drop_prob, \
                                    occ_dim=occ_dim, data_mean=data_mean)
            xi = np.repeat(xi, 2, axis=0)
            xo = np.repeat(xo, 2, axis=0)
            xm = np.repeat(xm, 2, axis=0)
            # draw some sample imputations from the model
            samp_count = xi.shape[0]
            _, model_samps = GPSI.sample_imputer(xi, xo, xm, use_guide_policy=False)
            seq_len = len(model_samps)
            seq_samps = np.zeros((seq_len*samp_count, model_samps[0].shape[1]))
            idx = 0
            for s1 in range(samp_count):
                for s2 in range(seq_len):
                    seq_samps[idx] = model_samps[s2][s1]
                    idx += 1
            file_name = "{0:s}_samples_ng_b{1:d}.png".format(result_tag, i)
            utils.visualize_samples(seq_samps, file_name, num_rows=20)
            # show KLds and NLLs on a step-by-step basis
            xb = to_fX( Xva[0:1000] )
            xi, xo, xm = construct_masked_data(xb, drop_prob=drop_prob, \
                                    occ_dim=occ_dim, data_mean=data_mean)
            step_costs = GPSI.compute_per_step_cost(xi, xo, xm)
            step_nlls = step_costs[0]
            step_klds = step_costs[1]
            step_nums = np.arange(step_nlls.shape[0])
            file_name = "{0:s}_NLL_b{1:d}.png".format(result_tag, i)
            utils.plot_stem(step_nums, step_nlls, file_name)
            file_name = "{0:s}_KLD_b{1:d}.png".format(result_tag, i)
            utils.plot_stem(step_nums, step_klds, file_name)

#################################
#################################
## CHECK MNIST IMPUTER RESULTS ##
#################################
#################################

def test_mnist_results(step_type='add',
                       imp_steps=6,
                       occ_dim=15,
                       drop_prob=0.0):
    #########################################
    # Format the result tag more thoroughly #
    #########################################
    dp_int = int(100.0 * drop_prob)
    result_tag = "{}GPSI_OD{}_DP{}_IS{}_{}_NA".format(RESULT_PATH, occ_dim, dp_int, imp_steps, step_type)

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

    ##########################
    # Get some training data #
    ##########################
    # rng = np.random.RandomState(1234)
    # dataset = 'data/mnist.pkl.gz'
    # datasets = load_udm(dataset, as_shared=False, zero_mean=False)
    # Xtr = datasets[0][0]
    # Xva = datasets[1][0]
    # Xte = datasets[2][0]
    # # Merge validation set and training set, and test on test set.
    # #Xtr = np.concatenate((Xtr, Xva), axis=0)
    # #Xva = Xte
    # Xtr = to_fX(shift_and_scale_into_01(Xtr))
    # Xva = to_fX(shift_and_scale_into_01(Xva))
    # tr_samples = Xtr.shape[0]
    # va_samples = Xva.shape[0]
    batch_size = 250
    batch_reps = 1
    all_pix_mean = np.mean(np.mean(Xtr, axis=1))
    data_mean = to_fX( all_pix_mean * np.ones((Xtr.shape[1],)) )

    # Load parameters from a previously trained model
    print("Testing model load from file...")
    GPSI = load_gpsimputer_from_file(f_name="{}_PARAMS.pkl".format(result_tag), \
                                     rng=rng)

    ################################################################
    # Apply some updates, to check that they aren't totally broken #
    ################################################################
    log_name = "{}_FINAL_RESULTS_NEW.txt".format(result_tag)
    out_file = open(log_name, 'wb')

    Xva = row_shuffle(Xva)
    # record an estimate of performance on the test set
    str0 = "GUIDED SAMPLE BOUND:"
    print(str0)
    xi, xo, xm = construct_masked_data(Xva[:5000], drop_prob=drop_prob, \
                                       occ_dim=occ_dim, data_mean=data_mean)
    nll_0, kld_0 = GPSI.compute_fe_terms(xi, xo, xm, sample_count=10, \
                                         use_guide_policy=True)
    xi, xo, xm = construct_masked_data(Xva[5000:], drop_prob=drop_prob, \
                                       occ_dim=occ_dim, data_mean=data_mean)
    nll_1, kld_1 = GPSI.compute_fe_terms(xi, xo, xm, sample_count=10, \
                                         use_guide_policy=True)
    nll = np.concatenate((nll_0, nll_1))
    kld = np.concatenate((kld_0, kld_1))
    vfe = np.mean(nll) + np.mean(kld)
    str1 = "    va_nll_bound : {}".format(vfe)
    str2 = "    va_nll_term  : {}".format(np.mean(nll))
    str3 = "    va_kld_q2p   : {}".format(np.mean(kld))
    joint_str = "\n".join([str0, str1, str2, str3])
    print(joint_str)
    out_file.write(joint_str+"\n")
    out_file.flush()
    # record an estimate of performance on the test set
    str0 = "UNGUIDED SAMPLE BOUND:"
    print(str0)
    xi, xo, xm = construct_masked_data(Xva[:5000], drop_prob=drop_prob, \
                                       occ_dim=occ_dim, data_mean=data_mean)
    nll_0, kld_0 = GPSI.compute_fe_terms(xi, xo, xm, sample_count=10, \
                                         use_guide_policy=False)
    xi, xo, xm = construct_masked_data(Xva[5000:], drop_prob=drop_prob, \
                                       occ_dim=occ_dim, data_mean=data_mean)
    nll_1, kld_1 = GPSI.compute_fe_terms(xi, xo, xm, sample_count=10, \
                                         use_guide_policy=False)
    nll = np.concatenate((nll_0, nll_1))
    kld = np.concatenate((kld_0, kld_1))
    str1 = "    va_nll_bound : {}".format(np.mean(nll))
    str2 = "    va_nll_term  : {}".format(np.mean(nll))
    str3 = "    va_kld_q2p   : {}".format(np.mean(kld))
    joint_str = "\n".join([str0, str1, str2, str3])
    print(joint_str)
    out_file.write(joint_str+"\n")
    out_file.flush()

if __name__=="__main__":
    #########
    # MNIST #
    #########
    # TRAINING
    #test_mnist(step_type='add', occ_dim=14, drop_prob=0.0)
    #test_mnist(step_type='add', occ_dim=16, drop_prob=0.0)
    #test_mnist(step_type='add', occ_dim=0, drop_prob=0.6)
    #test_mnist(step_type='add', occ_dim=0, drop_prob=0.8)
    #test_mnist(step_type='jump', occ_dim=14, drop_prob=0.0)
    #test_mnist(step_type='jump', occ_dim=16, drop_prob=0.0)
    #test_mnist(step_type='jump', occ_dim=0, drop_prob=0.6)
    #test_mnist(step_type='jump', occ_dim=0, drop_prob=0.8)
    #test_mnist(step_type='add', imp_steps=1, occ_dim=0, drop_prob=0.9)
    #test_mnist(step_type='add', imp_steps=2, occ_dim=0, drop_prob=0.9)
    test_mnist(step_type='add', imp_steps=5, occ_dim=0, drop_prob=0.9)
    #test_mnist(step_type='add', imp_steps=10, occ_dim=0, drop_prob=0.9)
    #test_mnist(step_type='add', imp_steps=15, occ_dim=0, drop_prob=0.9)
    #test_mnist(step_type='add', imp_steps=10, occ_dim=0, drop_prob=1.0)

    # RESULTS
    # test_mnist_results(step_type='add', occ_dim=14, drop_prob=0.0)
    # test_mnist_results(step_type='add', occ_dim=16, drop_prob=0.0)
    # test_mnist_results(step_type='add', occ_dim=0, drop_prob=0.6)
    # test_mnist_results(step_type='add', occ_dim=0, drop_prob=0.7)
    # test_mnist_results(step_type='add', occ_dim=0, drop_prob=0.8)
    # test_mnist_results(step_type='add', occ_dim=0, drop_prob=0.9)
    # test_mnist_results(step_type='jump', occ_dim=14, drop_prob=0.0)
    # test_mnist_results(step_type='jump', occ_dim=16, drop_prob=0.0)
    # test_mnist_results(step_type='jump', occ_dim=0, drop_prob=0.6)
    # test_mnist_results(step_type='jump', occ_dim=0, drop_prob=0.7)
    # test_mnist_results(step_type='jump', occ_dim=0, drop_prob=0.8)
    # test_mnist_results(step_type='jump', occ_dim=0, drop_prob=0.9)
    #test_mnist_results(step_type='add', imp_steps=1, occ_dim=0, drop_prob=0.9)
    #test_mnist_results(step_type='add', imp_steps=2, occ_dim=0, drop_prob=0.9)
    #test_mnist_results(step_type='add', imp_steps=5, occ_dim=0, drop_prob=0.9)
    #test_mnist_results(step_type='add', imp_steps=10, occ_dim=0, drop_prob=0.9)
    #test_mnist_results(step_type='add', imp_steps=15, occ_dim=0, drop_prob=0.9)
