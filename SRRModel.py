#############################################################################
# Code for a variational Sequential Revelation and Refinement Model.        #
#############################################################################

# basic python
import cPickle
import numpy as np
import numpy.random as npr
from collections import OrderedDict
import numexpr as ne

# theano business
import theano
import theano.tensor as T
#from theano.tensor.shared_randomstreams import RandomStreams as RandStream
from theano.sandbox.cuda.rng_curand import CURAND_RandomStreams as RandStream

# phil's sweetness
from DKCode import get_adam_updates, get_adadelta_updates
from LogPDFs import log_prob_bernoulli, log_prob_gaussian2, gaussian_kld
from HelperFuncs import to_fX

##############################################
# IMPLEMENTATION FOR A THING THAT DOES STUFF #
##############################################
#                                            #
# This thing does cool stuff, very deeply!   #
##############################################

class SRRModel(object):
    """
    Controller for training a sequential revelation and refinement model.

    Parameters:
        rng: numpy.random.RandomState (for reproducibility)
        x_out: the goal state for iterative refinement
        p_zi_given_xi: InfNet for stochastic part of step
        p_sip1_given_zi: HydraNet for deterministic part of step
        p_x_given_si: HydraNet for transform from s-space to x-space
        q_zi_given_xi: InfNet for the guide policy
        params: REQUIRED PARAMS SHOWN BELOW
                x_dim: dimension of observations to construct
                z_dim: dimension of latent space for policy wobble
                s_dim: dimension of space in which to perform construction
                use_p_x_given_si: boolean for whether to use ----
                init_steps: number of refinement steps prior to any revelations
                reveal_steps: number of times to make revelations
                refine_steps: number of refinement steps per revelation
                reveal_rate: percentage of values to reveal at each revelation
                step_type: either "add" or "jump"
                x_type: can be "bernoulli" or "gaussian"
                obs_transform: can be 'none' or 'sigmoid'
    """
    def __init__(self, rng=None,
            x_out=None, \
            p_zi_given_xi=None, \
            p_sip1_given_zi=None, \
            p_x_given_si=None, \
            q_zi_given_xi=None, \
            params=None, \
            shared_param_dicts=None):
        # setup a rng for this SRRModel
        self.rng = RandStream(rng.randint(100000))

        # grab the user-provided parameters
        self.params = params
        self.x_dim = self.params['x_dim']
        self.z_dim = self.params['z_dim']
        self.s_dim = self.params['s_dim']
        self.use_p_x_given_si = self.params['use_p_x_given_si']
        self.init_steps = self.params['init_steps']
        self.reveal_steps = self.params['reveal_steps']
        self.refine_steps = self.params['refine_steps']
        self.reveal_rate = self.params['reveal_rate']
        self.step_type = self.params['step_type']
        self.x_type = self.params['x_type']
        nice_nums = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
        assert(self.init_steps in nice_nums)
        assert(self.reveal_steps in nice_nums)
        assert(self.refine_steps in nice_nums)
        assert((self.x_type == 'bernoulli') or (self.x_type == 'gaussian'))
        assert((self.step_type == 'add') or (self.step_type == 'jump'))
        if self.use_p_x_given_si:
            print("Constructing hypotheses indirectly in s-space...")
        else:
            print("Constructing hypotheses directly in x-space...")
            assert(self.s_dim == self.x_dim)
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

        # grab handles to the relevant networks
        self.p_zi_given_xi = p_zi_given_xi
        self.p_sip1_given_zi = p_sip1_given_zi
        self.p_x_given_si = p_x_given_si
        self.q_zi_given_xi = q_zi_given_xi

        # record the symbolic variables that will provide inputs to the
        # computation graph created to describe this SRRModel
        self.x_out = x_out
        self.zi_zmuv = T.tensor3()   # ZMUV gauss noise for policy wobble
        self.rev_masks = T.tensor3() # masks indicating values to reveal
        self.total_steps = self.init_steps + \
                           (self.reveal_steps * self.refine_steps)

        # setup switching variable for changing between sampling/training
        zero_ary = to_fX( np.zeros((1,)) )
        self.train_switch = theano.shared(value=zero_ary, name='srrm_train_switch')
        self.set_train_switch(1.0)

        if self.shared_param_dicts is None:
            # initialize the parameters "owned" by this model
            s0_init = to_fX( np.zeros((self.s_dim,)) )
            self.s0 = theano.shared(value=s0_init, name='srrm_s0')
            self.obs_logvar = theano.shared(value=zero_ary, name='srrm_obs_logvar')
            self.bounded_logvar = 8.0 * T.tanh((1.0/8.0) * self.obs_logvar[0])
            self.shared_param_dicts = {}
            self.shared_param_dicts['s0'] = self.s0
            self.shared_param_dicts['obs_logvar'] = self.obs_logvar
        else:
            # grab the parameters required by this model from a given dict
            self.s0 = self.shared_param_dicts['s0']
            self.obs_logvar = self.shared_param_dicts['obs_logvar']
            self.bounded_logvar = 8.0 * T.tanh((1.0/8.0) * self.obs_logvar[0])

        ##################################################################
        # Setup the sequential revelation and refinement loop using scan #
        ##################################################################
        # zi_zmuv: This is a sequence of ZMUV gaussian samples that will be
        #          reparametrized to get the samples from the various relevant
        #          conditional distributions
        #
        # rev_masks: This is a sequence of "unmasking" masks. When one of these
        #            masking variables is 1, the corresponding value in
        #            self.x_out will be "revealed" to the generator. Prediction
        #            error is measured for a value _every_ time it is revealed.
        #            Once revealed, a value remains "visible" to the generator.
        #            So, for each trial of the generator, each value in the
        #            goal state should be revealed at most once. At the final
        #            step of each trial, prediction error is measured on all
        #            values not yet revealed to the generator.
        #
        # si: This is the current "belief state" for each trial in the training
        #     batch. The belief state is updated in each iteration, and passed
        #     forward through the recurrence.
        #
        # mi: This is the current mask for revealed/unrevealed values for each
        #     trial in the training batch. revealed == 1 and unrevealed == 0.
        #     I.e. this mask tells which values we're allowed to condition on
        #     while updating the belief state for each trial.
        #
        def srr_step_func(zi_zmuv, rev_masks, si, mi):
            # transform the current belief state into an observation
            si_as_x = self._from_si_to_x(si)
            # get the full unmasked observation and gradient of the NLL of the
            # full observation based on the current belief state
            xi_full = self.x_out
            grad_full = xi_full - si_as_x
            # get the masked observation and masked reconstruction gradient
            # based on the set of revealed values
            xi_masked = (mi * xi_full) + \
                        ((1.0 - mi) * si_as_x)
            grad_masked = grad_full * mi

            # get samples of next zi, according to the primary policy
            zi_p_mean, zi_p_logvar = self.p_zi_given_xi.apply( \
                    T.horizontal_stack(xi_masked, grad_masked), \
                    do_samples=False)
            zi_p = zi_p_mean + (T.exp(0.5 * zi_p_logvar) * zi_zmuv)
            # get samples of next zi, according to the guide policy
            zi_q_mean, zi_q_logvar = self.p_zi_given_xi.apply( \
                    T.horizontal_stack(xi_masked, grad_full), \
                    do_samples=False)
            zi_q = zi_q_mean + (T.exp(0.5 * zi_q_logvar) * zi_zmuv)
            # make zi samples that can be switched between zi_p and zi_q
            zi = ((self.train_switch[0] * zi_q) + \
                 ((1.0 - self.train_switch[0]) * zi_p))

            # compute relevant KLds for this step
            kldi_q2p = gaussian_kld(zi_q_mean, zi_q_logvar, \
                                    zi_p_mean, zi_p_logvar) # KL(q || p)
            kldi_p2q = gaussian_kld(zi_p_mean, zi_p_logvar, \
                                    zi_q_mean, zi_q_logvar) # KL(p || q)
            kldi_p2g = gaussian_kld(zi_p_mean, zi_p_logvar, \
                                    0.0, 0.0) # KL(p || N(0, I))

            # compute next si, given sampled zi (i.e. update the belief state)
            hydra_out = self.p_sip1_given_zi.apply(zi)
            si_step = hydra_out[0]
            if (self.step_type == 'jump'):
                # jump steps always do a full swap (like standard VAE)
                sip1 = si_step
            else:
                # additive steps adjust the current guesses incrementally
                write_gate = T.nnet.sigmoid(2.0 + hydra_out[1])
                erase_gate = T.nnet.sigmoid(2.0 + hydra_out[2])
                # LSTM-style update
                sip1 = (erase_gate * si) + (write_gate * si_step)
            # record which values will be revealed for the first time -- these
            # values will be available for conditioning at the next step.
            newly_revealed = (1.0 - mi) * rev_masks
            # update the revelation mask, which is 1 for all values which
            # will be available for conditioning in the next iteration
            mip1 = mi + newly_revealed
            # compute NLL only for the newly revealed values
            nlli = self._construct_nll_costs(sip1, self.x_out, newly_revealed)
            # each loop iteration produces the following values:
            #   sip1: belief state at end of current step
            #   mip1: revealed values mask to use in next step
            #   nlli: NLL for values revealed at end of current step
            #   kldi_q2p: KL(q || p) for the current step
            #   kldi_p2q: KL(p || q) for the current step
            #   kldi_p2g: KL(p || N(0,I)) for the current step
            return sip1, mip1, nlli, kldi_q2p, kldi_p2q, kldi_p2g

        # initialize belief state to self.s0
        self.s0_full = T.alloc(0.0, self.x_out.shape[0], self.s_dim) + self.s0
        # initialize revelation masks to 0 for all values in all trials
        self.m0_full = T.zeros_like(self.x_out)
        # setup initial values to pass to scan op
        outputs_init = [self.s0_full, self.m0_full, None, None, None, None]
        sequences_init = [self.zi_zmuv, self.rev_masks]
        # apply scan op for the sequential imputation loop
        self.scan_results, self.scan_updates = theano.scan(srr_step_func, \
                    outputs_info=outputs_init, \
                    sequences=sequences_init)

        # grab results of the scan op. all values are computed for each step
        self.si = self.scan_results[0]       # belief states
        self.mi = self.scan_results[1]       # revelation masks
        self.nlli = self.scan_results[2]     # NLL on newly revealed values
        self.kldi_q2p = self.scan_results[3] # KL(q || p)
        self.kldi_p2q = self.scan_results[4] # KL(p || q)
        self.kldi_p2g = self.scan_results[5] # KL(p || N(0,I))

        ######################################################################
        # ALL SYMBOLIC VARS NEEDED FOR THE OBJECTIVE SHOULD NOW BE AVAILABLE #
        ######################################################################

        # shared var learning rate for generator and inferencer
        zero_ary = to_fX( np.zeros((1,)) )
        self.lr = theano.shared(value=zero_ary, name='srr_lr')
        # shared var momentum parameters for generator and inferencer
        self.mom_1 = theano.shared(value=zero_ary, name='srr_mom_1')
        self.mom_2 = theano.shared(value=zero_ary, name='srr_mom_2')
        # init parameters for controlling learning dynamics
        self.set_sgd_params()
        # init shared var for weighting prior kld against reconstruction
        self.lam_kld_p = theano.shared(value=zero_ary, name='srr_lam_kld_p')
        self.lam_kld_q = theano.shared(value=zero_ary, name='srr_lam_kld_q')
        self.lam_kld_g = theano.shared(value=zero_ary, name='srr_lam_kld_g')
        self.set_lam_kld(lam_kld_p=0.05, lam_kld_q=0.95, lam_kld_g=0.0)
        # init shared var for controlling l2 regularization on params
        self.lam_l2w = theano.shared(value=zero_ary, name='srr_lam_l2w')
        self.set_lam_l2w(1e-5)

        # Grab all of the "optimizable" parameters in "group 1"
        self.joint_params = [self.s0, self.obs_logvar]
        self.joint_params.extend(self.p_zi_given_xi.mlp_params)
        self.joint_params.extend(self.p_sip1_given_zi.mlp_params)
        self.joint_params.extend(self.p_x_given_si.mlp_params)
        self.joint_params.extend(self.q_zi_given_xi.mlp_params)

        #################################
        # CONSTRUCT THE KLD-BASED COSTS #
        #################################
        self.kld_p, self.kld_q, self.kld_g = self._construct_kld_costs(p=1.0)
        self.kld_costs = (self.lam_kld_p[0] * self.kld_p) + \
                         (self.lam_kld_q[0] * self.kld_q) + \
                         (self.lam_kld_g[0] * self.kld_g)
        self.kld_cost = T.mean(self.kld_costs)
        #################################
        # CONSTRUCT THE NLL-BASED COSTS #
        #################################
        self.nll_costs = T.sum(self.nlli, axis=0) # sum the per-step NLLs
        self.nll_cost = T.mean(self.nll_costs)
        self.nll_bounds = self.nll_costs.ravel() + self.kld_q.ravel()
        self.nll_bound = T.mean(self.nll_bounds)
        ########################################
        # CONSTRUCT THE REST OF THE JOINT COST #
        ########################################
        param_reg_cost = self._construct_reg_costs()
        self.reg_cost = self.lam_l2w[0] * param_reg_cost
        self.joint_cost = self.nll_cost + self.kld_cost + self.reg_cost
        ##############################
        # CONSTRUCT A PER-TRIAL COST #
        ##############################
        self.obs_costs = self.nll_costs + self.kld_costs

        # Get the gradient of the joint cost for all optimizable parameters
        print("Computing gradients of self.joint_cost...")
        self.joint_grads = OrderedDict()
        grad_list = T.grad(self.joint_cost, self.joint_params)
        for i, p in enumerate(self.joint_params):
            self.joint_grads[p] = grad_list[i]

        # Construct the updates for the generator and inferencer networks
        self.joint_updates = get_adam_updates(params=self.joint_params, \
                grads=self.joint_grads, alpha=self.lr, \
                beta1=self.mom_1, beta2=self.mom_2, \
                mom2_init=1e-3, smoothing=1e-5, max_grad_norm=10.0)
        for k, v in self.scan_updates.items():
            self.joint_updates[k] = v

        # Construct theano functions for training and diagnostic computations
        print("Compiling sequence sampler...")
        self.sequence_sampler = self._construct_sequence_sampler()
        print("Compiling cost computer...")
        self.compute_raw_costs = self._construct_raw_costs()
        print("Compiling training function...")
        self.train_joint = self._construct_train_joint()
        print("Compiling free-energy sampler...")
        self.compute_fe_terms = self._construct_compute_fe_terms()
        # make easy access points for some interesting parameters
        #self.gen_inf_weights = self.p_zi_given_xi.shared_layers[0].W
        return

    def _from_si_to_x(self, si):
        """
        Convert the given si from s-space to x-space.
        """
        if self.use_p_x_given_si:
            x_pre_trans, _ = self.p_x_given_si.apply(si)
        else:
            x_pre_trans = si
        x_post_trans = self.obs_transform(x_pre_trans)
        return x_post_trans

    def set_sgd_params(self, lr=0.01, mom_1=0.9, mom_2=0.999):
        """
        Set learning rate and momentum parameter for all updates.
        """
        zero_ary = np.zeros((1,))
        # set learning rate
        new_lr = zero_ary + lr
        self.lr.set_value(to_fX(new_lr))
        # set momentums (use first and second order "momentum")
        new_mom_1 = zero_ary + mom_1
        self.mom_1.set_value(to_fX(new_mom_1))
        new_mom_2 = zero_ary + mom_2
        self.mom_2.set_value(to_fX(new_mom_2))
        return

    def set_lam_kld(self, lam_kld_p=0.0, lam_kld_q=1.0, lam_kld_g=0.0):
        """
        Set the relative weight of prior KL-divergence vs. data likelihood.
        """
        zero_ary = np.zeros((1,))
        new_lam = zero_ary + lam_kld_p
        self.lam_kld_p.set_value(to_fX(new_lam))
        new_lam = zero_ary + lam_kld_q
        self.lam_kld_q.set_value(to_fX(new_lam))
        new_lam = zero_ary + lam_kld_g
        self.lam_kld_g.set_value(to_fX(new_lam))
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

    def _construct_zi_zmuv(self, xo):
        """
        Construct the necessary ZMUV gaussian samples for generating
        trajectories from this SRRModel, for input matrix xo.
        """
        zi_zmuv = self.rng.normal( \
                size=(self.total_steps, xo.shape[0], self.z_dim), \
                avg=0.0, std=1.0, dtype=theano.config.floatX)
        return zi_zmuv

    def _construct_rev_masks(self, xo):
        """
        Compute the sequential revelation masks for the input batch in xo.
        """
        # Generate independently sampled masks for each revelation step
        rand_vals = self.rng.uniform( \
                size=(self.reveal_steps, xo.shape[0], xo.shape[1]), \
                low=0.0, high=1.0, dtype=theano.config.floatX)
        rand_masks = rand_vals < self.reveal_rate
        # Repeat each revelation mask for some number of refinement steps
        middle_masks = rand_masks.repeat(self.refine_steps, axis=0)
        # Make a final mask that reveals all values in all observations
        final_mask = T.alloc(1.0, 1, xo.shape[0], xo.shape[1])
        if self.init_steps == 1:
            # Values are revealed at the _end_ of each scan step, so the first
            # step where values are revealed acts as an init step.
            mask_list = [middle_masks, final_mask]
        else:
            # To perform multiple init steps, we can front-pad the sequence of
            # revelation masks with masks that don't reveal any values.
            init_masks = T.alloc(0.0, self.init_steps-1, xo.shape[0], xo.shape[1])
            mask_list = [init_masks, middle_masks, final_mask]
        # Concatenate masks for all steps
        rev_masks = T.concatenate(mask_list, axis=0)
        return rev_masks

    def _construct_nll_costs(self, si, xo, nll_mask):
        """
        Construct the negative log-likelihood part of free energy.
        -- only check NLL where nll_mask == 1
        """
        # average log-likelihood over the refinement sequence
        xh = self._from_si_to_x( si )
        if self.x_type == 'bernoulli':
            ll_costs = log_prob_bernoulli(xo, xh, mask=nll_mask)
        else:
            ll_costs = log_prob_gaussian2(xo, xh, \
                    log_vars=self.bounded_logvar, mask=nll_mask)
        nll_costs = -ll_costs.flatten()
        return nll_costs

    def _construct_kld_costs(self, p=1.0):
        """
        Construct the policy KL-divergence part of cost to minimize.
        """
        kld_pis = []
        kld_qis = []
        kld_gis = []
        for i in range(self.total_steps):
            kld_pis.append(T.sum(self.kldi_p2q[i]**p, axis=1))
            kld_qis.append(T.sum(self.kldi_q2p[i]**p, axis=1))
            kld_gis.append(T.sum(self.kldi_p2g[i]**p, axis=1))
        # compute the batch-wise costs
        kld_pi = sum(kld_pis)
        kld_qi = sum(kld_qis)
        kld_gi = sum(kld_gis)
        return [kld_pi, kld_qi, kld_gi]

    def _construct_reg_costs(self):
        """
        Construct the cost for low-level basic regularization. E.g. for
        applying l2 regularization to the network activations and parameters.
        """
        param_reg_cost = sum([T.sum(p**2.0) for p in self.joint_params])
        return param_reg_cost

    def _construct_compute_fe_terms(self):
        """
        Construct a function for computing terms in variational free energy.
        """
        # setup some symbolic variables for theano to deal with
        xo = T.matrix()
        zizmuv = self._construct_zi_zmuv(xo)
        revmasks = self._construct_rev_masks(xo)
        # construct values to output
        nll = self.nll_costs.flatten()
        kld = self.kld_q.flatten()
        # compile theano function for a one-sample free-energy estimate
        fe_term_sample = theano.function(inputs=[ xo ], \
                outputs=[nll, kld], \
                givens={self.x_out: xo, \
                        self.zi_zmuv: zizmuv, \
                        self.rev_masks: revmasks}, \
                updates=self.scan_updates, \
                on_unused_input='ignore')
        # construct a wrapper function for multi-sample free-energy estimate
        def fe_term_estimator(XO, sample_count=20, use_guide_policy=True):
            # set model to desired generation mode
            old_switch = self.train_switch.get_value(borrow=False)
            if use_guide_policy:
                # take samples from the guide policy
                self.set_train_switch(switch_val=1.0)
            else:
                # take samples from the primary policy
                self.set_train_switch(switch_val=0.0)
            # compute a multi-sample estimate of variational free-energy
            nll_sum = np.zeros((XO.shape[0],))
            kld_sum = np.zeros((XO.shape[0],))
            for i in range(sample_count):
                result = fe_term_sample(XO)
                nll_sum += result[0].ravel()
                kld_sum += result[1].ravel()
            mean_nll = nll_sum / float(sample_count)
            mean_kld = kld_sum / float(sample_count)
            # set model back to either training or generation mode
            self.set_train_switch(switch_val=old_switch)
            if not use_guide_policy:
                # no KLd if samples are from the primary policy...
                mean_kld = 0.0 * mean_kld
            return [mean_nll, mean_kld]
        return fe_term_estimator

    def _construct_raw_costs(self):
        """
        Construct all the raw, i.e. not weighted by any lambdas, costs.
        """
        # setup some symbolic variables for theano to deal with
        xo = T.matrix()
        zizmuv = self._construct_zi_zmuv(xo)
        revmasks = self._construct_rev_masks(xo)
        # compile theano function for computing the costs
        all_step_costs = [self.nlli, self.kldi_q2p, self.kldi_p2q, self.kldi_p2g]
        cost_func = theano.function(inputs=[ xo ], \
                    outputs=all_step_costs, \
                    givens={self.x_out: xo, \
                            self.zi_zmuv: zizmuv, \
                            self.rev_masks: revmasks}, \
                    updates=self.scan_updates, \
                    on_unused_input='ignore')
        # make a function for computing multi-sample estimates of cost
        def raw_cost_computer(XO):
            _all_costs = cost_func(to_fX(XO))
            _kld_q2p = np.sum(np.mean(_all_costs[1], axis=1, keepdims=True), axis=0)
            _kld_p2q = np.sum(np.mean(_all_costs[2], axis=1, keepdims=True), axis=0)
            _kld_p2g = np.sum(np.mean(_all_costs[3], axis=1, keepdims=True), axis=0)
            _step_klds = np.mean(np.sum(_all_costs[1], axis=2, keepdims=True), axis=1)
            _step_klds = to_fX( np.asarray([k for k in _step_klds]) )
            _step_nlls = np.mean(_all_costs[0], axis=1)
            _step_nlls = to_fX( np.asarray([k for k in _step_nlls]) )
            results = [_step_nlls, _step_klds, _kld_q2p, _kld_p2q, _kld_p2g]
            return results
        return raw_cost_computer

    def _construct_train_joint(self):
        """
        Construct theano function to train all networks jointly.
        """
        # setup some symbolic variables for theano to deal with
        xo = T.matrix()
        zizmuv = self._construct_zi_zmuv(xo)
        revmasks = self._construct_rev_masks(xo)
        # collect the outputs to return from this function
        outputs = [self.joint_cost, self.nll_bound, self.nll_cost, \
                   self.kld_cost, self.reg_cost, self.obs_costs]
        # compile the theano function
        func = theano.function(inputs=[ xo ], \
                outputs=outputs, \
                givens={self.x_out: xo, \
                        self.zi_zmuv: zizmuv, \
                        self.rev_masks: revmasks}, \
                updates=self.joint_updates, \
                on_unused_input='ignore')
        return func

    def _construct_sequence_sampler(self):
        """
        Construct theano function to train all networks jointly.
        """
        # setup some symbolic variables for theano to deal with
        xo = T.matrix()
        zizmuv = self._construct_zi_zmuv(xo)
        revmasks = self._construct_rev_masks(xo)
        # collect the outputs to return from this function
        states = [self._from_si_to_x(self.s0_full)] + \
                 [self._from_si_to_x(self.si[i]) for i in range(self.total_steps)]
        masks = [self.m0_full] + [self.mi[i] for i in range(self.total_steps)]
        outputs = states + masks
        # compile the theano function
        func = theano.function(inputs=[ xo ], \
                outputs=outputs, \
                givens={self.x_out: xo, \
                        self.zi_zmuv: zizmuv, \
                        self.rev_masks: revmasks}, \
                updates=self.joint_updates, \
                on_unused_input='ignore')
        # visualize trajectories generated by the model
        def sample_func(XO, use_guide_policy=False):
            # set model to desired generation mode
            old_switch = self.train_switch.get_value(borrow=False)
            if use_guide_policy:
                # take samples from the guide policy
                self.set_train_switch(switch_val=1.0)
            else:
                # take samples from the primary policy
                self.set_train_switch(switch_val=0.0)
            # get belief states and masks generated by the scan loop
            scan_vals = func(to_fX(XO))
            step_count = self.total_steps + 1
            seq_shape = (step_count, XO.shape[0], XO.shape[1])
            xm_seq = np.zeros(seq_shape).astype(theano.config.floatX)
            xi_seq = np.zeros(seq_shape).astype(theano.config.floatX)
            mi_seq = np.zeros(seq_shape).astype(theano.config.floatX)
            for i in range(step_count):
                _xi = scan_vals[i]
                _mi = scan_vals[i + step_count]
                _xm = (_mi * XO) + ((1.0 - _mi) * _xi)
                xm_seq[i,:,:] = _xm
                xi_seq[i,:,:] = _xi
                mi_seq[i,:,:] = _mi
            # set model back to either training or generation mode
            self.set_train_switch(switch_val=old_switch)
            return [xm_seq, xi_seq, mi_seq]
        return sample_func

    def save_to_file(self, f_name=None):
        """
        Dump important stuff to a Python pickle, so that we can reload this
        model later.
        """
        assert(not (f_name is None))
        f_handle = file(f_name, 'wb')
        # dump the dict self.params, which just holds "simple" python values
        cPickle.dump(self.params, f_handle, protocol=-1)
        # make a copy of self.shared_param_dicts, with numpy arrays in place
        # of the theano shared variables
        numpy_param_dicts = {}
        for key in self.shared_param_dicts:
            numpy_ary = self.shared_param_dicts[key].get_value(borrow=False)
            numpy_param_dicts[key] = numpy_ary
        # dump the numpy version of self.shared_param_dicts to pickle file
        cPickle.dump(numpy_param_dicts, f_handle, protocol=-1)
        # get numpy dicts for each of the "child" models that we must save
        child_model_dicts = {}
        child_model_dicts['p_zi_given_xi'] = self.p_zi_given_xi.save_to_dict()
        child_model_dicts['p_sip1_given_zi'] = self.p_sip1_given_zi.save_to_dict()
        child_model_dicts['p_x_given_si'] = self.p_x_given_si.save_to_dict()
        child_model_dicts['q_zi_given_xi'] = self.q_zi_given_xi.save_to_dict()
        # dump the numpy child model dicts to the pickle file
        cPickle.dump(child_model_dicts, f_handle, protocol=-1)
        f_handle.close()
        return

def load_srrmodel_from_file(f_name=None, rng=None):
    """
    Load a clone of some previously trained model.
    """
    from InfNet import load_infnet_from_dict
    from HydraNet import load_hydranet_from_dict
    assert(not (f_name is None))
    pickle_file = open(f_name)
    # reload the basic python parameters
    self_dot_params = cPickle.load(pickle_file)
    # reload the theano shared parameters
    self_dot_numpy_param_dicts = cPickle.load(pickle_file)
    self_dot_shared_param_dicts = {}
    for key in self_dot_numpy_param_dicts:
        val = to_fX(self_dot_numpy_param_dicts[key])
        self_dot_shared_param_dicts[key] = theano.shared(val)
    # reload the child models
    child_model_dicts = cPickle.load(pickle_file)
    xd = T.matrix()
    p_zi_given_xi = load_infnet_from_dict( \
            child_model_dicts['p_zi_given_xi'], rng=rng, Xd=xd)
    p_sip1_given_zi = load_hydranet_from_dict( \
            child_model_dicts['p_sip1_given_zi'], rng=rng, Xd=xd)
    p_x_given_si = load_hydranet_from_dict( \
            child_model_dicts['p_x_given_si'], rng=rng, Xd=xd)
    q_zi_given_xi = load_infnet_from_dict( \
            child_model_dicts['q_zi_given_xi'], rng=rng, Xd=xd)
    # now, create a new SRRModel based on the loaded data
    xo = T.matrix()
    clone_net = SRRModel(rng=rng, \
                         x_out=xo, \
                         p_zi_given_xi=p_zi_given_xi, \
                         p_sip1_given_zi=p_sip1_given_zi, \
                         p_x_given_si=p_x_given_si, \
                         q_zi_given_xi=q_zi_given_xi, \
                         params=self_dot_params, \
                         shared_param_dicts=self_dot_shared_param_dicts)
    # helpful output
    print("==================================================")
    print("LOADED SRRModel WITH PARAMS:")
    for k in self_dot_params:
        print("    {0:s}: {1:s}".format(str(k), str(self_dot_params[k])))
    print("==================================================")
    return clone_net



if __name__=="__main__":
    print("Hello world!")







##############
# EYE BUFFER #
##############
