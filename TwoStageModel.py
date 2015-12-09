#############################################################################
# Code for managing and training a variational Iterative Refinement Model.  #
#############################################################################

# basic python
import numpy as np
import numpy.random as npr
from collections import OrderedDict

# theano business
import theano
import theano.tensor as T
#from theano.tensor.shared_randomstreams import RandomStreams as RandStream
from theano.sandbox.cuda.rng_curand import CURAND_RandomStreams as RandStream

# phil's sweetness
from NetLayers import HiddenLayer, DiscLayer, relu_actfun, softplus_actfun
from HelperFuncs import to_fX, reparametrize
from DKCode import get_adam_updates, get_adadelta_updates
from LogPDFs import log_prob_bernoulli, log_prob_gaussian2, gaussian_kld

####################
# IMPLEMENTATATION #
####################

class TwoStageModel1(object):
    """
    Controller for training a two-step hierarchical generative model.
      x: the "observation" variables
      z: the "prior" latent variables
      h: the "hidden" latent variables

    Generative model is: p(x) = \sum_{z,h} p(x|h) p(h|z) p(z)
    Variational model is: q(h,z|x) = q(h|z,x) q(z|x)

    Parameters:
        rng: numpy.random.RandomState (for reproducibility)
        x_in: the input data to encode
        x_out: the target output to decode
        p_h_given_z: HydraNet for h given z (2 outputs)
        p_x_given_h: HydraNet for x given h (2 outputs)
        q_z_given_x: HydraNet for z given x (2 outputs)
        q_h_given_z_x: HydraNet for h given z and x (2 outputs)
        x_dim: dimension of the "observation" space
        z_dim: dimension of the "prior" latent space
        h_dim: dimension of the "hidden" latent space
        params: REQUIRED PARAMS SHOWN BELOW
                x_type: can be "bernoulli" or "gaussian"
                obs_transform: can be 'none' or 'sigmoid'
    """
    def __init__(self, rng=None,
            x_in=None, x_out=None,
            p_h_given_z=None,
            p_x_given_h=None,
            q_z_given_x=None,
            q_h_given_z_x=None,
            x_dim=None,
            z_dim=None,
            h_dim=None,
            h_det_dim=None,
            params=None,
            shared_param_dicts=None):
        # setup a rng for this GIPair
        self.rng = RandStream(rng.randint(100000))

        # grab the user-provided parameters
        self.params = params
        self.x_type = self.params['x_type']
        assert((self.x_type == 'bernoulli') or (self.x_type == 'gaussian'))
        if 'obs_transform' in self.params:
            assert((self.params['obs_transform'] == 'sigmoid') or \
                    (self.params['obs_transform'] == 'none'))
            if self.params['obs_transform'] == 'sigmoid':
                self.obs_transform = lambda x: T.nnet.sigmoid(x)
            else:
                self.obs_transform = lambda x: x
        else:
            self.obs_transform = lambda x: T.nnet.sigmoid(x)
        if self.x_type == 'bernoulli':
            self.obs_transform = lambda x: T.nnet.sigmoid(x)
        self.shared_param_dicts = shared_param_dicts

        # record the dimensions of various spaces relevant to this model
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.h_dim = h_dim
        self.h_det_dim = h_det_dim

        # grab handles to the relevant HydraNets
        self.q_z_given_x = q_z_given_x
        self.q_h_given_z_x = q_h_given_z_x
        self.p_h_given_z = p_h_given_z
        self.p_x_given_h = p_x_given_h

        # record the symbolic variables that will provide inputs to the
        # computation graph created to describe this MultiStageModel
        self.x_in = x_in
        self.x_out = x_out

        # setup switching variable for changing between sampling/training
        zero_ary = to_fX( np.zeros((1,)) )
        self.train_switch = theano.shared(value=zero_ary, name='tsm_train_switch')
        self.set_train_switch(1.0)

        if self.shared_param_dicts is None:
            # initialize "optimizable" parameters specific to this MSM
            init_vec = to_fX( np.zeros((1,self.z_dim)) )
            self.p_z_mean = theano.shared(value=init_vec, name='tsm_p_z_mean')
            self.p_z_logvar = theano.shared(value=init_vec, name='tsm_p_z_logvar')
            self.obs_logvar = theano.shared(value=zero_ary, name='tsm_obs_logvar')
            self.bounded_logvar = 8.0 * T.tanh((1.0/8.0) * self.obs_logvar)
            self.shared_param_dicts = {}
            self.shared_param_dicts['p_z_mean'] = self.p_z_mean
            self.shared_param_dicts['p_z_logvar'] = self.p_z_logvar
            self.shared_param_dicts['obs_logvar'] = self.obs_logvar
        else:
            self.p_z_mean = self.shared_param_dicts['p_z_mean']
            self.p_z_logvar = self.shared_param_dicts['p_z_logvar']
            self.obs_logvar = self.shared_param_dicts['obs_logvar']
            self.bounded_logvar = 8.0 * T.tanh((1.0/8.0) * self.obs_logvar)

        ##############################################
        # Setup the TwoStageModels main computation. #
        ##############################################
        print("Building TSM...")
        # samples of "hidden" latent state (from both p and q)
        z_q_mean, z_q_logvar = self.q_z_given_x.apply(self.x_in)
        z_q = reparametrize(z_q_mean, z_q_logvar, rng=self.rng)

        z_p_mean = self.p_z_mean.repeat(z_q.shape[0], axis=0)
        z_p_logvar = self.p_z_logvar.repeat(z_q.shape[0], axis=0)
        z_p = reparametrize(z_p_mean, z_p_logvar, rng=self.rng)

        self.z = (self.train_switch[0] * z_q) + \
                 ((1.0 - self.train_switch[0]) * z_p)
        # compute relevant KLds for this step
        self.kld_z_q2p = gaussian_kld(z_q_mean, z_q_logvar,
                                      z_p_mean, z_p_logvar)
        self.kld_z_p2q = gaussian_kld(z_p_mean, z_p_logvar,
                                      z_q_mean, z_q_logvar)
        # samples of "hidden" latent state (from both p and q)
        h_p_mean, h_p_logvar = self.p_h_given_z.apply(self.z)
        h_p = reparametrize(h_p_mean, h_p_logvar, rng=self.rng)

        h_q_mean, h_q_logvar = self.q_h_given_z_x.apply(
                T.concatenate([h_p_mean, self.x_out], axis=1))
        h_q = reparametrize(h_q_mean, h_q_logvar, rng=self.rng)

        # compute "stochastic" and "deterministic" parts of latent state
        h_sto = (self.train_switch[0] * h_q) + \
                ((1.0 - self.train_switch[0]) * h_p)
        h_det = h_p_mean
        if self.h_det_dim is None:
            # don't pass forward any deterministic state
            self.h = h_sto
        else:
            # pass forward some deterministic state
            self.h = T.concatenate([h_det[:,:self.h_det_dim],
                                    h_sto[:,self.h_det_dim:]], axis=1)
        # compute relevant KLds for this step
        self.kld_h_q2p = gaussian_kld(h_q_mean, h_q_logvar,
                                      h_p_mean, h_p_logvar)
        self.kld_h_p2q = gaussian_kld(h_p_mean, h_p_logvar,
                                      h_q_mean, h_q_logvar)

        # p_x_given_h generates an observation x conditioned on the "hidden"
        # latent variables h.
        self.x_gen, _ = self.p_x_given_h.apply(self.h)

        ######################################################################
        # ALL SYMBOLIC VARS NEEDED FOR THE OBJECTIVE SHOULD NOW BE AVAILABLE #
        ######################################################################

        # shared var learning rate for generator and inferencer
        zero_ary = to_fX( np.zeros((1,)) )
        self.lr = theano.shared(value=zero_ary, name='tsm_lr')
        # shared var momentum parameters for generator and inferencer
        self.mom_1 = theano.shared(value=zero_ary, name='tsm_mom_1')
        self.mom_2 = theano.shared(value=zero_ary, name='tsm_mom_2')
        # init parameters for controlling learning dynamics
        self.set_sgd_params()
        # init shared var for weighting nll of data given posterior sample
        self.lam_nll = theano.shared(value=zero_ary, name='tsm_lam_nll')
        self.set_lam_nll(lam_nll=1.0)
        # init shared var for weighting prior kld against reconstruction
        self.lam_kld_q2p = theano.shared(value=zero_ary, name='tsm_lam_kld_q2p')
        self.lam_kld_p2q = theano.shared(value=zero_ary, name='tsm_lam_kld_p2q')
        self.set_lam_kld(lam_kld_q2p=1.0, lam_kld_p2q=0.0)
        # init shared var for controlling l2 regularization on params
        self.lam_l2w = theano.shared(value=zero_ary, name='tsm_lam_l2w')
        self.set_lam_l2w(1e-5)

        # get optimizable parameters belonging to the TwoStageModel
        self_params = [self.obs_logvar] #+ [self.p_z_mean, self.p_z_logvar]
        # get optimizable parameters belonging to the underlying networks
        child_params = []
        child_params.extend(self.q_z_given_x.mlp_params)
        child_params.extend(self.q_h_given_z_x.mlp_params)
        child_params.extend(self.p_h_given_z.mlp_params)
        child_params.extend(self.p_x_given_h.mlp_params)
        # make a joint list of all optimizable parameters
        self.joint_params = self_params + child_params

        #################################
        # CONSTRUCT THE KLD-BASED COSTS #
        #################################
        self.kld_z = (self.lam_kld_q2p[0] * self.kld_z_q2p) + \
                     (self.lam_kld_p2q[0] * self.kld_z_p2q)
        self.kld_h = (self.lam_kld_q2p[0] * self.kld_h_q2p) + \
                     (self.lam_kld_p2q[0] * self.kld_h_p2q)
        self.kld_costs = T.sum(self.kld_z, axis=1) + \
                         T.sum(self.kld_h, axis=1)
        # compute "mean" (rather than per-input) costs
        self.kld_cost = T.mean(self.kld_costs)
        #################################
        # CONSTRUCT THE NLL-BASED COSTS #
        #################################
        self.nll_costs = self._construct_nll_costs(self.x_out)
        self.nll_cost = self.lam_nll[0] * T.mean(self.nll_costs)
        ########################################
        # CONSTRUCT THE REST OF THE JOINT COST #
        ########################################
        param_reg_cost = self._construct_reg_costs()
        self.reg_cost = self.lam_l2w[0] * param_reg_cost
        self.joint_cost = self.nll_cost + self.kld_cost + self.reg_cost
        ##############################
        # CONSTRUCT A PER-INPUT COST #
        ##############################
        self.obs_costs = self.nll_costs + self.kld_costs

        # get the gradient of the joint cost for all optimizable parameters
        print("Computing gradients of self.joint_cost...")
        self.joint_grads = OrderedDict()
        grad_list = T.grad(self.joint_cost, self.joint_params)
        for i, p in enumerate(self.joint_params):
            self.joint_grads[p] = grad_list[i]

        # construct the updates for the generator and inferencer networks
        all_updates = get_adam_updates(params=self.joint_params,
                grads=self.joint_grads, alpha=self.lr,
                beta1=self.mom_1, beta2=self.mom_2,
                mom2_init=1e-3, smoothing=1e-4, max_grad_norm=5.0)
        self.joint_updates = OrderedDict()
        for k in all_updates:
            self.joint_updates[k] = all_updates[k]

        # Construct a function for jointly training the generator/inferencer
        print("Compiling training function...")
        self.train_joint = self._construct_train_joint()
        print("Compiling free-energy sampler...")
        self.compute_fe_terms = self._construct_compute_fe_terms()
        print("Compiling open-loop model sampler...")
        self.sample_from_prior = self._construct_sample_from_prior()
        return

    def set_sgd_params(self, lr=0.01, mom_1=0.9, mom_2=0.999):
        """
        Set learning rate and momentum parameter for all updates.
        """
        zero_ary = np.zeros((1,))
        # set learning rate
        new_lr = zero_ary + lr
        self.lr.set_value(to_fX(new_lr))
        # set momentums
        new_mom_1 = zero_ary + mom_1
        self.mom_1.set_value(to_fX(new_mom_1))
        new_mom_2 = zero_ary + mom_2
        self.mom_2.set_value(to_fX(new_mom_2))
        return

    def set_lam_nll(self, lam_nll=1.0):
        """
        Set weight for controlling the influence of the data likelihood.
        """
        zero_ary = np.zeros((1,))
        new_lam = zero_ary + lam_nll
        self.lam_nll.set_value(to_fX(new_lam))
        return

    def set_lam_kld(self, lam_kld_q2p=1.0, lam_kld_p2q=1.0):
        """
        Set the relative weight of various KL-divergences.
        """
        zero_ary = np.zeros((1,))
        new_lam = zero_ary + lam_kld_q2p
        self.lam_kld_q2p.set_value(to_fX(new_lam))
        new_lam = zero_ary + lam_kld_p2q
        self.lam_kld_p2q.set_value(to_fX(new_lam))
        return

    def set_lam_l2w(self, lam_l2w=1e-3):
        """
        Set the relative strength of l2 regularization on network params.
        """
        zero_ary = np.zeros((1,))
        new_lam = zero_ary + lam_l2w
        self.lam_l2w.set_value(to_fX(new_lam))
        return

    def set_train_switch(self, switch_val=0.0):
        """
        Set the switch for changing between training and sampling behavior.
        """
        if (switch_val < 0.5):
            switch_val = 0.0
        else:
            switch_val = 1.0
        zero_ary = np.zeros((1,))
        new_val = zero_ary + switch_val
        self.train_switch.set_value(to_fX(new_val))
        return

    def _construct_nll_costs(self, xo):
        """
        Construct the negative log-likelihood part of free energy.
        """
        # average log-likelihood over the refinement sequence
        xh = self.obs_transform(self.x_gen)
        if self.x_type == 'bernoulli':
            ll_costs = log_prob_bernoulli(xo, xh)
        else:
            ll_costs = log_prob_gaussian2(xo, xh, log_vars=self.bounded_logvar)
        nll_costs = -ll_costs
        return nll_costs

    def _construct_kld_costs(self, p=1.0):
        """
        Construct the posterior KL-divergence part of cost to minimize.
        """
        kld_z_q2p = T.sum(self.kld_z_q2p**p, axis=1, keepdims=True)
        kld_z_p2q = T.sum(self.kld_z_p2q**p, axis=1, keepdims=True)
        kld_h_q2p = T.sum(self.kld_h_q2p**p, axis=1, keepdims=True)
        kld_h_p2q = T.sum(self.kld_h_p2q**p, axis=1, keepdims=True)
        return [kld_z_q2p, kld_z_p2q, kld_h_q2p, kld_h_p2q]

    def _construct_reg_costs(self):
        """
        Construct the cost for low-level basic regularization. E.g. for
        applying l2 regularization to the network activations and parameters.
        """
        param_reg_cost = sum([T.sum(p**2.0) for p in self.joint_params])
        return param_reg_cost

    def _construct_train_joint(self):
        """
        Construct theano function to train all networks jointly.
        """
        # setup some symbolic variables for theano to deal with
        xi = T.matrix()
        xo = T.matrix()
        br = T.lscalar()
        nll = self.nll_costs
        kld_z = T.sum(self.kld_z_q2p, axis=1)
        kld_h = T.sum(self.kld_h_q2p, axis=1)
        # collect the outputs to return from this function
        outputs = [self.joint_cost, self.nll_cost, self.kld_cost,
                   self.reg_cost, nll, kld_z, kld_h]
        # compile the theano function
        func = theano.function(inputs=[ xi, xo, br ],
                outputs=outputs,
                givens={ self.x_in: xi.repeat(br, axis=0),
                         self.x_out: xo.repeat(br, axis=0) },
                updates=self.joint_updates)
        return func

    def _construct_compute_fe_terms(self):
        """
        Construct a function for computing terms in variational free energy.
        """
        # construct values to output
        nll = self._construct_nll_costs(self.x_out)
        kld_z = self.kld_z_q2p
        kld_h = self.kld_h_q2p
        # compile theano function for a one-sample free-energy estimate
        fe_term_sample = theano.function(inputs=[self.x_in, self.x_out],
                                         outputs=[nll, kld_z, kld_h])
        # construct a wrapper function for multi-sample free-energy estimate
        def fe_term_estimator(XI, XO, sample_count):
            # compute a multi-sample estimate of variational free-energy
            nll_sum = np.zeros((XI.shape[0],))
            kld_z_sum = np.zeros((XI.shape[0],))
            kld_h_sum = np.zeros((XI.shape[0],))
            for i in range(sample_count):
                result = fe_term_sample(XI, XO)
                nll_sum += result[0].ravel()
                kld_z_sum += np.sum(result[1], axis=1).ravel()
                kld_h_sum += np.sum(result[2], axis=1).ravel()
            mean_nll = nll_sum / float(sample_count)
            mean_kld = (kld_z_sum + kld_h_sum) / float(sample_count)
            mean_kld_z = kld_z_sum / float(sample_count)
            mean_kld_h = kld_h_sum / float(sample_count)
            return [mean_nll, mean_kld, mean_kld_z, mean_kld_h]
        return fe_term_estimator

    def _construct_sample_from_prior(self):
        """
        Construct a function for drawing independent samples from the
        distribution generated by this TwoStageModel.
        """
        x_sym = T.matrix()
        sample_func = theano.function(inputs=[x_sym],
                outputs=self.obs_transform(self.x_gen),
                givens={self.x_in: T.zeros_like(x_sym),
                        self.x_out: T.zeros_like(x_sym)})
        def prior_sampler(samp_count):
            x_samps = to_fX( np.zeros((samp_count, self.x_dim)) )
            old_switch = self.train_switch.get_value(borrow=False)
            # set model to generation mode
            self.set_train_switch(switch_val=0.0)
            # generate samples from model
            model_samps = sample_func(x_samps)
            # set model back to previous mode
            self.set_train_switch(switch_val=old_switch)
            return model_samps
        return prior_sampler

class TwoStageModel2(object):
    """
    Controller for training a two-step hierarchical generative model.
      x: the "observation" variables
      z: the "prior" latent variables
      h: the "hidden" latent variables

    Generative model is: p(x) = \sum_{z,h} p(x|h) p(h|z) p(z)
    Variational model is: q(h,z|x) = q(h|x) q(z|h)

    Parameters:
        rng: numpy.random.RandomState (for reproducibility)
        x_in: the input data to encode
        x_out: the target output to decode
        p_h_given_z: HydraNet for h given z (2 outputs)
        p_x_given_h: HydraNet for x given h (2 outputs)
        q_h_given_x: HydraNet for h given x (2 outputs)
        q_z_given_h: HydraNet for z given h (2 outputs)
        x_dim: dimension of the "observation" space
        z_dim: dimension of the "prior" latent space
        h_dim: dimension of the "hidden" latent space
        params: REQUIRED PARAMS SHOWN BELOW
                x_type: can be "bernoulli" or "gaussian"
                obs_transform: can be 'none' or 'sigmoid'
    """
    def __init__(self, rng=None,
            x_in=None, x_out=None,
            p_h_given_z=None,
            p_x_given_h=None,
            q_h_given_x=None,
            q_z_given_h=None,
            x_dim=None,
            z_dim=None,
            h_dim=None,
            params=None,
            shared_param_dicts=None):
        # setup a rng for this GIPair
        self.rng = RandStream(rng.randint(100000))

        # grab the user-provided parameters
        self.params = params
        self.x_type = self.params['x_type']
        assert((self.x_type == 'bernoulli') or (self.x_type == 'gaussian'))
        if 'obs_transform' in self.params:
            assert((self.params['obs_transform'] == 'sigmoid') or \
                    (self.params['obs_transform'] == 'none'))
            if self.params['obs_transform'] == 'sigmoid':
                self.obs_transform = lambda x: T.nnet.sigmoid(x)
            else:
                self.obs_transform = lambda x: x
        else:
            self.obs_transform = lambda x: T.nnet.sigmoid(x)
        if self.x_type == 'bernoulli':
            self.obs_transform = lambda x: T.nnet.sigmoid(x)
        self.shared_param_dicts = shared_param_dicts

        # record the dimensions of various spaces relevant to this model
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.h_dim = h_dim

        # grab handles to the relevant HydraNets
        self.q_h_given_x = q_h_given_x
        self.q_z_given_h = q_z_given_h
        self.p_h_given_z = p_h_given_z
        self.p_x_given_h = p_x_given_h

        # record the symbolic variables that will provide inputs to the
        # computation graph created to describe this MultiStageModel
        self.x_in = x_in
        self.x_out = x_out

        # setup switching variable for changing between sampling/training
        zero_ary = to_fX( np.zeros((1,)) )
        self.train_switch = theano.shared(value=zero_ary, name='tsm_train_switch')
        self.set_train_switch(1.0)

        if self.shared_param_dicts is None:
            # initialize "optimizable" parameters specific to this TSM
            init_vec = to_fX( np.zeros((1,self.z_dim)) )
            self.p_z_mean = theano.shared(value=init_vec, name='tsm_p_z_mean')
            self.p_z_logvar = theano.shared(value=init_vec, name='tsm_p_z_logvar')
            self.obs_logvar = theano.shared(value=zero_ary, name='tsm_obs_logvar')
            self.bounded_logvar = 8.0 * T.tanh((1.0/8.0) * self.obs_logvar)
            self.shared_param_dicts = {}
            self.shared_param_dicts['p_z_mean'] = self.p_z_mean
            self.shared_param_dicts['p_z_logvar'] = self.p_z_logvar
            self.shared_param_dicts['obs_logvar'] = self.obs_logvar
        else:
            self.p_z_mean = self.shared_param_dicts['p_z_mean']
            self.p_z_logvar = self.shared_param_dicts['p_z_logvar']
            self.obs_logvar = self.shared_param_dicts['obs_logvar']
            self.bounded_logvar = 8.0 * T.tanh((1.0/8.0) * self.obs_logvar)

        ##############################################
        # Setup the TwoStageModels main computation. #
        ##############################################
        print("Building TSM...")
        # samples of "hidden" latent state (from q)
        h_q_mean, h_q_logvar = self.q_h_given_x.apply(self.x_in)
        h_q = reparametrize(h_q_mean, h_q_logvar, rng=self.rng)
        # samples of "prior" latent state (from q)
        z_q_mean, z_q_logvar = self.q_z_given_h.apply(h_q)
        z_q = reparametrize(z_q_mean, z_q_logvar, rng=self.rng)
        # samples of "prior" latent state (from p)
        z_p_mean = self.p_z_mean.repeat(z_q.shape[0], axis=0)
        z_p_logvar = self.p_z_logvar.repeat(z_q.shape[0], axis=0)
        z_p = reparametrize(z_p_mean, z_p_logvar, rng=self.rng)
        # samples from z -- switched between q/p
        self.z = (self.train_switch[0] * z_q) + \
                 ((1.0 - self.train_switch[0]) * z_p)
        # samples of "hidden" latent state (from p)
        h_p_mean, h_p_logvar = self.p_h_given_z.apply(self.z)
        h_p = reparametrize(h_p_mean, h_p_logvar, rng=self.rng)
        # samples from h -- switched between q/p
        self.h = (self.train_switch[0] * h_q) + \
                 ((1.0 - self.train_switch[0]) * h_p)
        # compute KLds for "prior" and "hidden" latent distributions
        self.kld_z_q2p = log_prob_gaussian2(self.z, z_q_mean, z_q_logvar) - \
                         log_prob_gaussian2(self.z, z_p_mean, z_p_logvar)
        self.kld_h_q2p = log_prob_gaussian2(self.h, h_q_mean, h_q_logvar) - \
                         log_prob_gaussian2(self.h, h_p_mean, h_p_logvar)
        self.kld_z_p2q = self.kld_z_q2p
        self.kld_h_p2q = self.kld_h_q2p

        # p_x_given_h generates an observation x conditioned on the "hidden"
        # latent variables h.
        self.x_gen, _ = self.p_x_given_h.apply(self.h)


        ######################################################################
        # ALL SYMBOLIC VARS NEEDED FOR THE OBJECTIVE SHOULD NOW BE AVAILABLE #
        ######################################################################

        # shared var learning rate for generator and inferencer
        zero_ary = to_fX( np.zeros((1,)) )
        self.lr = theano.shared(value=zero_ary, name='tsm_lr')
        # shared var momentum parameters for generator and inferencer
        self.mom_1 = theano.shared(value=zero_ary, name='tsm_mom_1')
        self.mom_2 = theano.shared(value=zero_ary, name='tsm_mom_2')
        # init parameters for controlling learning dynamics
        self.set_sgd_params()
        # init shared var for weighting nll of data given posterior sample
        self.lam_nll = theano.shared(value=zero_ary, name='tsm_lam_nll')
        self.set_lam_nll(lam_nll=1.0)
        # init shared var for weighting prior kld against reconstruction
        self.lam_kld_q2p = theano.shared(value=zero_ary, name='tsm_lam_kld_q2p')
        self.lam_kld_p2q = theano.shared(value=zero_ary, name='tsm_lam_kld_p2q')
        self.set_lam_kld(lam_kld_q2p=1.0, lam_kld_p2q=0.0)
        # init shared var for controlling l2 regularization on params
        self.lam_l2w = theano.shared(value=zero_ary, name='tsm_lam_l2w')
        self.set_lam_l2w(1e-5)

        # get optimizable parameters belonging to the TwoStageModel
        self_params = [self.obs_logvar] #+ [self.p_z_mean, self.p_z_logvar]
        # get optimizable parameters belonging to the underlying networks
        child_params = []
        child_params.extend(self.q_h_given_x.mlp_params)
        child_params.extend(self.q_z_given_h.mlp_params)
        child_params.extend(self.p_h_given_z.mlp_params)
        child_params.extend(self.p_x_given_h.mlp_params)
        # make a joint list of all optimizable parameters
        self.joint_params = self_params + child_params


        #################################
        # CONSTRUCT THE KLD-BASED COSTS #
        #################################
        self.kld_z = (self.lam_kld_q2p[0] * self.kld_z_q2p) + \
                     (self.lam_kld_p2q[0] * self.kld_z_p2q)
        self.kld_h = (self.lam_kld_q2p[0] * self.kld_h_q2p) + \
                     (self.lam_kld_p2q[0] * self.kld_h_p2q)
        self.kld_costs = T.sum(self.kld_z, axis=1) + \
                         T.sum(self.kld_h, axis=1)
        # compute "mean" (rather than per-input) costs
        self.kld_cost = T.mean(self.kld_costs)
        #################################
        # CONSTRUCT THE NLL-BASED COSTS #
        #################################
        self.nll_costs = self._construct_nll_costs(self.x_out)
        self.nll_cost = self.lam_nll[0] * T.mean(self.nll_costs)
        ########################################
        # CONSTRUCT THE REST OF THE JOINT COST #
        ########################################
        param_reg_cost = self._construct_reg_costs()
        self.reg_cost = self.lam_l2w[0] * param_reg_cost
        self.joint_cost = self.nll_cost + self.kld_cost + self.reg_cost
        ##############################
        # CONSTRUCT A PER-INPUT COST #
        ##############################
        self.obs_costs = self.nll_costs + self.kld_costs

        # get the gradient of the joint cost for all optimizable parameters
        print("Computing gradients of self.joint_cost...")
        self.joint_grads = OrderedDict()
        grad_list = T.grad(self.joint_cost, self.joint_params)
        for i, p in enumerate(self.joint_params):
            self.joint_grads[p] = grad_list[i]

        # construct the updates for the generator and inferencer networks
        all_updates = get_adam_updates(params=self.joint_params,
                grads=self.joint_grads, alpha=self.lr,
                beta1=self.mom_1, beta2=self.mom_2,
                mom2_init=1e-3, smoothing=1e-4, max_grad_norm=5.0)
        self.joint_updates = OrderedDict()
        for k in all_updates:
            self.joint_updates[k] = all_updates[k]

        # Construct a function for jointly training the generator/inferencer
        print("Compiling training function...")
        self.train_joint = self._construct_train_joint()
        print("Compiling free-energy sampler...")
        self.compute_fe_terms = self._construct_compute_fe_terms()
        print("Compiling open-loop model sampler...")
        self.sample_from_prior = self._construct_sample_from_prior()
        return

    def set_sgd_params(self, lr=0.01, mom_1=0.9, mom_2=0.999):
        """
        Set learning rate and momentum parameter for all updates.
        """
        zero_ary = np.zeros((1,))
        # set learning rate
        new_lr = zero_ary + lr
        self.lr.set_value(to_fX(new_lr))
        # set momentums
        new_mom_1 = zero_ary + mom_1
        self.mom_1.set_value(to_fX(new_mom_1))
        new_mom_2 = zero_ary + mom_2
        self.mom_2.set_value(to_fX(new_mom_2))
        return

    def set_lam_nll(self, lam_nll=1.0):
        """
        Set weight for controlling the influence of the data likelihood.
        """
        zero_ary = np.zeros((1,))
        new_lam = zero_ary + lam_nll
        self.lam_nll.set_value(to_fX(new_lam))
        return

    def set_lam_kld(self, lam_kld_q2p=1.0, lam_kld_p2q=1.0):
        """
        Set the relative weight of various KL-divergences.
        """
        zero_ary = np.zeros((1,))
        new_lam = zero_ary + lam_kld_q2p
        self.lam_kld_q2p.set_value(to_fX(new_lam))
        new_lam = zero_ary + lam_kld_p2q
        self.lam_kld_p2q.set_value(to_fX(new_lam))
        return

    def set_lam_l2w(self, lam_l2w=1e-3):
        """
        Set the relative strength of l2 regularization on network params.
        """
        zero_ary = np.zeros((1,))
        new_lam = zero_ary + lam_l2w
        self.lam_l2w.set_value(to_fX(new_lam))
        return

    def set_train_switch(self, switch_val=0.0):
        """
        Set the switch for changing between training and sampling behavior.
        """
        if (switch_val < 0.5):
            switch_val = 0.0
        else:
            switch_val = 1.0
        zero_ary = np.zeros((1,))
        new_val = zero_ary + switch_val
        self.train_switch.set_value(to_fX(new_val))
        return

    def _construct_nll_costs(self, xo):
        """
        Construct the negative log-likelihood part of free energy.
        """
        # average log-likelihood over the refinement sequence
        xh = self.obs_transform(self.x_gen)
        if self.x_type == 'bernoulli':
            ll_costs = log_prob_bernoulli(xo, xh)
        else:
            ll_costs = log_prob_gaussian2(xo, xh, log_vars=self.bounded_logvar)
        nll_costs = -ll_costs
        return nll_costs

    def _construct_kld_costs(self, p=1.0):
        """
        Construct the posterior KL-divergence part of cost to minimize.
        """
        kld_z_q2p = T.sum(self.kld_z_q2p**p, axis=1, keepdims=True)
        kld_z_p2q = T.sum(self.kld_z_p2q**p, axis=1, keepdims=True)
        kld_h_q2p = T.sum(self.kld_h_q2p**p, axis=1, keepdims=True)
        kld_h_p2q = T.sum(self.kld_h_p2q**p, axis=1, keepdims=True)
        return [kld_z_q2p, kld_z_p2q, kld_h_q2p, kld_h_p2q]

    def _construct_reg_costs(self):
        """
        Construct the cost for low-level basic regularization. E.g. for
        applying l2 regularization to the network activations and parameters.
        """
        param_reg_cost = sum([T.sum(p**2.0) for p in self.joint_params])
        return param_reg_cost

    def _construct_train_joint(self):
        """
        Construct theano function to train all networks jointly.
        """
        # setup some symbolic variables for theano to deal with
        xi = T.matrix()
        xo = T.matrix()
        br = T.lscalar()
        #
        nll = self.nll_costs
        kldz = T.sum(self.kld_z, axis=1)
        kldh = T.sum(self.kld_h, axis=1)
        # collect the outputs to return from this function
        outputs = [self.joint_cost, self.nll_cost, self.kld_cost,
                   self.reg_cost, self.nll_costs, kldz, kldh]
        # compile the theano function
        func = theano.function(inputs=[ xi, xo, br ],
                outputs=outputs,
                givens={ self.x_in: xi.repeat(br, axis=0),
                         self.x_out: xo.repeat(br, axis=0) },
                updates=self.joint_updates)
        return func

    def _construct_compute_fe_terms(self):
        """
        Construct a function for computing terms in variational free energy.
        """
        # construct values to output
        nll = self._construct_nll_costs(self.x_out)
        kld_z = self.kld_z_q2p
        kld_h = self.kld_h_q2p
        # compile theano function for a one-sample free-energy estimate
        fe_term_sample = theano.function(inputs=[self.x_in, self.x_out],
                                         outputs=[nll, kld_z, kld_h])
        # construct a wrapper function for multi-sample free-energy estimate
        def fe_term_estimator(XI, XO, sample_count):
            # compute a multi-sample estimate of variational free-energy
            nll_sum = np.zeros((XI.shape[0],))
            kld_z_sum = np.zeros((XI.shape[0],))
            kld_h_sum = np.zeros((XI.shape[0],))
            for i in range(sample_count):
                result = fe_term_sample(XI, XO)
                nll_sum += result[0].ravel()
                kld_z_sum += np.sum(result[1], axis=1).ravel()
                kld_h_sum += np.sum(result[2], axis=1).ravel()
            mean_nll = nll_sum / float(sample_count)
            mean_kld = (kld_z_sum + kld_h_sum) / float(sample_count)
            return [mean_nll, mean_kld]
        return fe_term_estimator

    def _construct_sample_from_prior(self):
        """
        Construct a function for drawing independent samples from the
        distribution generated by this TwoStageModel.
        """
        x_sym = T.matrix()
        sample_func = theano.function(inputs=[x_sym],
                outputs=self.obs_transform(self.x_gen),
                givens={self.x_in: T.zeros_like(x_sym),
                        self.x_out: T.zeros_like(x_sym)})
        def prior_sampler(samp_count):
            x_samps = to_fX( np.zeros((samp_count, self.x_dim)) )
            old_switch = self.train_switch.get_value(borrow=False)
            # set model to generation mode
            self.set_train_switch(switch_val=0.0)
            # generate samples from model
            model_samps = sample_func(x_samps)
            # set model back to previous mode
            self.set_train_switch(switch_val=old_switch)
            return model_samps
        return prior_sampler


if __name__=="__main__":
    print("Hello world!")







##############
# EYE BUFFER #
##############
