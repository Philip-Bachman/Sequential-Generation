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
from SRRModel import SRRModel, load_srrmodel_from_file
from load_data import load_udm, load_tfd, load_svhn_gray, load_binarized_mnist
from HelperFuncs import construct_masked_data, shift_and_scale_into_01, \
                        row_shuffle, to_fX

RESULT_PATH = "SRRM_RESULTS/"

###############################
###############################
## TEST GPS IMPUTER ON MNIST ##
###############################
###############################

def test_mnist(step_type='add', \
               init_steps=3, \
               reveal_steps=7, \
               refine_steps=1, \
               reveal_rate=0.2):
    #########################################
    # Format the result tag more thoroughly #
    #########################################
    result_tag = "{}SRRM_IN{}_RV{}_RF{}_RT{}_ST{}".format(RESULT_PATH, \
            init_steps, reveal_steps, refine_steps, reveal_rate, step_type)

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
    batch_size = 250

    ############################################################
    # Setup some parameters for the Iterative Refinement Model #
    ############################################################
    x_dim = Xtr.shape[1]
    s_dim = x_dim
    #s_dim = 300
    z_dim = 100
    init_scale = 1.0

    x_out_sym = T.matrix('x_out_sym')

    #################
    # p_zi_given_xi #
    #################
    params = {}
    shared_config = [(x_dim + x_dim), 500, 500]
    top_config = [shared_config[-1], z_dim]
    params['shared_config'] = shared_config
    params['mu_config'] = top_config
    params['sigma_config'] = top_config
    params['activation'] = relu_actfun
    params['init_scale'] = init_scale
    params['vis_drop'] = 0.0
    params['hid_drop'] = 0.0
    params['bias_noise'] = 0.0
    params['input_noise'] = 0.0
    params['build_theano_funcs'] = False
    p_zi_given_xi = InfNet(rng=rng, Xd=x_out_sym, \
            params=params, shared_param_dicts=None)
    p_zi_given_xi.init_biases(0.2)
    ###################
    # p_sip1_given_zi #
    ###################
    params = {}
    shared_config = [z_dim, 500, 500]
    output_config = [s_dim, s_dim, s_dim]
    params['shared_config'] = shared_config
    params['output_config'] = output_config
    params['activation'] = relu_actfun
    params['init_scale'] = init_scale
    params['vis_drop'] = 0.0
    params['hid_drop'] = 0.0
    params['bias_noise'] = 0.0
    params['input_noise'] = 0.0
    params['build_theano_funcs'] = False
    p_sip1_given_zi = HydraNet(rng=rng, Xd=x_out_sym, \
            params=params, shared_param_dicts=None)
    p_sip1_given_zi.init_biases(0.2)
    ################
    # p_x_given_si #
    ################
    params = {}
    shared_config = [s_dim, 500]
    output_config = [x_dim, x_dim]
    params['shared_config'] = shared_config
    params['output_config'] = output_config
    params['activation'] = relu_actfun
    params['init_scale'] = init_scale
    params['vis_drop'] = 0.0
    params['hid_drop'] = 0.0
    params['bias_noise'] = 0.0
    params['input_noise'] = 0.0
    params['build_theano_funcs'] = False
    p_x_given_si = HydraNet(rng=rng, Xd=x_out_sym, \
            params=params, shared_param_dicts=None)
    p_x_given_si.init_biases(0.2)
    ###################
    # q_zi_given_xi #
    ###################
    params = {}
    shared_config = [(x_dim + x_dim), 500, 500]
    top_config = [shared_config[-1], z_dim]
    params['shared_config'] = shared_config
    params['mu_config'] = top_config
    params['sigma_config'] = top_config
    params['activation'] = relu_actfun
    params['init_scale'] = init_scale
    params['vis_drop'] = 0.0
    params['hid_drop'] = 0.0
    params['bias_noise'] = 0.0
    params['input_noise'] = 0.0
    params['build_theano_funcs'] = False
    q_zi_given_xi = InfNet(rng=rng, Xd=x_out_sym, \
            params=params, shared_param_dicts=None)
    q_zi_given_xi.init_biases(0.2)

    #########################################################
    # Define parameters for the SRRModel, and initialize it #
    #########################################################
    print("Building the SRRModel...")
    srrm_params = {}
    srrm_params['x_dim'] = x_dim
    srrm_params['z_dim'] = z_dim
    srrm_params['s_dim'] = s_dim
    srrm_params['use_p_x_given_si'] = False
    srrm_params['init_steps'] = init_steps
    srrm_params['reveal_steps'] = reveal_steps
    srrm_params['refine_steps'] = refine_steps
    srrm_params['reveal_rate'] = reveal_rate
    srrm_params['step_type'] = step_type
    srrm_params['x_type'] = 'bernoulli'
    srrm_params['obs_transform'] = 'sigmoid'
    SRRM = SRRModel(rng=rng, 
            x_out=x_out_sym, \
            p_zi_given_xi=p_zi_given_xi, \
            p_sip1_given_zi=p_sip1_given_zi, \
            p_x_given_si=p_x_given_si, \
            q_zi_given_xi=q_zi_given_xi, \
            params=srrm_params, \
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
        lam_scale = 1.0 - min(1.0, ((i+1) / 50000.0)) # decays from 1.0->0.0
        if (((i + 1) % 15000) == 0):
            learn_rate = learn_rate * 0.93
        if (i > 10000):
            momentum = 0.90
        else:
            momentum = 0.50
        # get the indices of training samples for this batch update
        batch_idx += batch_size
        if (np.max(batch_idx) >= tr_samples):
            # we finished an "epoch", so we rejumble the training set
            Xtr = row_shuffle(Xtr)
            batch_idx = np.arange(batch_size)
        # set sgd and objective function hyperparams for this update
        SRRM.set_sgd_params(lr=scale*learn_rate, \
                            mom_1=scale*momentum, mom_2=0.98)
        SRRM.set_train_switch(1.0)
        SRRM.set_lam_kld(lam_kld_p=0.05, lam_kld_q=0.95, lam_kld_g=(0.2 * lam_scale))
        SRRM.set_lam_l2w(1e-4)
        # perform a minibatch update and record the cost for this batch
        xb = to_fX( Xtr.take(batch_idx, axis=0) )
        result = SRRM.train_joint(xb)
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
            xb = Xva[0:5000]
            nll, kld = SRRM.compute_fe_terms(xb, sample_count=10)
            vfe = np.mean(nll) + np.mean(kld)
            str1 = "    va_nll_bound : {}".format(vfe)
            str2 = "    va_nll_term  : {}".format(np.mean(nll))
            str3 = "    va_kld_q2p   : {}".format(np.mean(kld))
            joint_str = "\n".join([str1, str2, str3])
            print(joint_str)
            out_file.write(joint_str+"\n")
            out_file.flush()
            # draw some sample imputations from the model
            xo = Xva[0:100]
            samp_count = xo.shape[0]
            xm_seq, xi_seq, mi_seq = SRRM.sequence_sampler(xo, use_guide_policy=False)
            seq_len = len(xm_seq)
            seq_samps = np.zeros((seq_len*samp_count, xm_seq[0].shape[1]))
            ######
            # xm #
            ######
            idx = 0
            for s1 in range(samp_count):
                for s2 in range(seq_len):
                    seq_samps[idx] = xm_seq[s2,s1,:]
                    idx += 1
            file_name = "{0:s}_xm_samples_b{1:d}.png".format(result_tag, i)
            utils.visualize_samples(seq_samps, file_name, num_rows=20)
            ######
            # xi #
            ######
            idx = 0
            for s1 in range(samp_count):
                for s2 in range(seq_len):
                    seq_samps[idx] = xi_seq[s2,s1,:]
                    idx += 1
            file_name = "{0:s}_xi_samples_b{1:d}.png".format(result_tag, i)
            utils.visualize_samples(seq_samps, file_name, num_rows=20)
            ######
            # mi #
            ######
            idx = 0
            for s1 in range(samp_count):
                for s2 in range(seq_len):
                    seq_samps[idx] = mi_seq[s2,s1,:]
                    idx += 1
            file_name = "{0:s}_mi_samples_b{1:d}.png".format(result_tag, i)
            utils.visualize_samples(seq_samps, file_name, num_rows=20)

if __name__=="__main__":
    #########
    # MNIST #
    #########
    # TRAINING
    test_mnist(step_type='add')