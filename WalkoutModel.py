#############################################################################
# Code for a variational Walkout model (kind of like a GSN).                #
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

class WalkoutModel(object):
    """
    Controller for training a forwards-backwards chainy model.

    Parameters:
        rng: numpy.random.RandomState (for reproducibility)
        x_out: the goal state for forwards-backwards walking process
        p_z_given_x: InfNet for stochastic part of step
        p_x_given_z: HydraNet for deterministic part of step
        params: REQUIRED PARAMS SHOWN BELOW
                x_dim: dimension of observations to construct
                z_dim: dimension of latent space for policy wobble
                walkout_steps: number of steps to walk out
                x_type: can be "bernoulli" or "gaussian"
                x_transform: can be 'none' or 'sigmoid'
    """
    def __init__(self, rng=None,
            x_out=None, \
            p_z_given_x=None, \
            p_x_given_z=None, \
            params=None, \
            shared_param_dicts=None):
        # setup a rng for this WalkoutModel
        self.rng = RandStream(rng.randint(100000))

        # grab the user-provided parameters
        self.params = params
        self.x_dim = self.params['x_dim']
        self.z_dim = self.params['z_dim']
        self.walkout_steps = self.params['walkout_steps']
        self.x_type = self.params['x_type']
        self.shared_param_dicts = shared_param_dicts
        if 'x_transform' in self.params:
            assert((self.params['x_transform'] == 'sigmoid') or \
                    (self.params['x_transform'] == 'none'))
            if self.params['x_transform'] == 'sigmoid':
                self.x_transform = lambda x: T.nnet.sigmoid(x)
            else:
                self.x_transform = lambda x: x
        else:
            self.x_transform = lambda x: T.nnet.sigmoid(x)
        if self.x_type == 'bernoulli':
            self.x_transform = lambda x: T.nnet.sigmoid(x)
        assert((self.x_type == 'bernoulli') or (self.x_type == 'gaussian'))
        assert((self.step_type == 'add') or (self.step_type == 'jump'))

        # grab handles to the relevant networks
        self.p_z_given_x = p_z_given_x
        self.p_x_given_z = p_x_given_z

        # record the symbolic variables that will provide inputs to the
        # computation graph created for this WalkoutModel
        self.x_out = x_out           # target output for generation
        self.zi_zmuv = T.tensor3()   # ZMUV gauss noise for walk-out wobble

        if self.shared_param_dicts is None:
            # initialize the parameters "owned" by this model
            zero_ary = to_fX( np.zeros((1,)) )
            self.obs_logvar = theano.shared(value=zero_ary, name='obs_logvar')
            self.bounded_logvar = 8.0 * T.tanh((1.0/8.0) * self.obs_logvar[0])
            self.shared_param_dicts = {}
            self.shared_param_dicts['obs_logvar'] = self.obs_logvar
        else:
            # grab the parameters required by this model from a given dict
            self.obs_logvar = self.shared_param_dicts['obs_logvar']
            self.bounded_logvar = 8.0 * T.tanh((1.0/8.0) * self.obs_logvar[0])

        ###############################################################
        # Setup the forwards (i.e. training) walk-out loop using scan #
        ###############################################################
        def forwards_loop(xi_zmuv, zi_zmuv, xi_fw, zi_fw):
            # get samples of next zi, according to the forwards model
            zi_fw_mean, zi_fw_logvar = self.p_z_given_x.apply(xi_fw, \
                                       do_samples=False)
            zi_fw = zi_fw_mean + (T.exp(0.5 * zi_fw_logvar) * zi_zmuv)

            # check reverse direction probability p(xi_fw | zi_fw)
            xi_bw_mean, xi_bw_logvar = self.p_x_given_z.apply(zi_fw, \
                                       do_samples=False)
            xi_bw_mean = self.x_transform(xi_bw_mean)
            nll_xi_bw = log_prob_gaussian2(xi_fw, xi_bw_mean, \
                        log_vars=xi_bw_logvar, mask=None)
            nll_xi_bw = nll_xi_bw.flatten()

            # get samples of next xi, according to the forwards model
            xi_fw_mean, xi_fw_logvar = self.p_x_given_z.apply(zi_fw, \
                                       do_samples=False)
            xi_fw_mean = self.x_transform(xi_fw_mean)
            xi_fw = xi_fw_mean + (T.exp(0.5 * xi_fw_logvar) * xi_zmuv)

            # check reverse direction probability p(zi_fw | xi_fw)
            zi_bw_mean, zi_bw_logvar = self.p_z_given_x.apply(xi_fw, \
                                       do_samples=False)
            nll_zi_bw = log_prob_gaussian2(zi_fw, zi_bw_mean, \
                        log_vars=zi_bw_logvar, mask=None)
            nll_zi_bw = nll_zi_bw.flatten()

            # each loop iteration produces the following values:
            #   xi_fw: xi generated fom zi by forwards walk
            #   zi_fw: zi generated fom xi by forwards walk
            #   xi_fw_mean: ----
            #   xi_fw_logvar: ----
            #   zi_fw_mean: ----
            #   zi_fw_logvar: ----
            #   nll_xi_bw: NLL for reverse step zi_fw -> xi_fw
            #   nll_zi_bw: NLL for reverse step xi_fw -> zi_fw
            return xi_fw, zi_fw, xi_fw_mean, xi_fw_logvar, zi_fw_mean, zi_fw_logvar, nll_xi_bw, nll_zi_bw

        # initialize states for x/z
        self.x0 = self.x_out
        self.z0 = T.alloc(0.0, self.x0.shape[0], self.z_dim)
        # setup initial values to pass to scan op
        outputs_init = [self.x0, self.z0, None, None, None, None, None, None]
        sequences_init = [self.xi_zmuv, self.zi_zmuv]
        # apply scan op for the sequential imputation loop
        self.scan_results, self.scan_updates = theano.scan(forwards_loop, \
                    outputs_info=outputs_init, \
                    sequences=sequences_init)

        # grab results of the scan op. all values are computed for each step
        self.xi = self.scan_results[0]
        self.zi = self.scan_results[1]
        self.xi_fw_mean = self.scan_results[2]
        self.xi_fw_logvar = self.scan_results[3]
        self.zi_fw_mean = self.scan_results[4]
        self.zi_fw_logvar = self.scan_results[5]
        self.nll_xi_bw = self.scan_results[6]
        self.nll_zi_bw = self.scan_results[7]

        ######################################################################
        # ALL SYMBOLIC VARS NEEDED FOR THE OBJECTIVE SHOULD NOW BE AVAILABLE #
        ######################################################################

        # shared var learning rate for generator and inferencer
        zero_ary = to_fX( np.zeros((1,)) )
        self.lr = theano.shared(value=zero_ary, name='srr_lr')
        # shared var momentum parameters for ADAM optimization
        self.mom_1 = theano.shared(value=zero_ary, name='srr_mom_1')
        self.mom_2 = theano.shared(value=zero_ary, name='srr_mom_2')
        # init parameters for controlling learning dynamics
        self.set_sgd_params()
        # init shared vars for weighting prior kld against reconstruction
        self.lam_kld_p = theano.shared(value=zero_ary, name='srr_lam_kld_p')
        self.lam_kld_q = theano.shared(value=zero_ary, name='srr_lam_kld_q')
        self.lam_kld_g = theano.shared(value=zero_ary, name='srr_lam_kld_g')
        self.lam_kld_s = theano.shared(value=zero_ary, name='srr_lam_kld_s')
        self.set_lam_kld(lam_kld_p=0.0, lam_kld_q=1.0, lam_kld_g=0.0, lam_kld_s=0.0)
        # init shared var for controlling l2 regularization on params
        self.lam_l2w = theano.shared(value=zero_ary, name='srr_lam_l2w')
        self.set_lam_l2w(1e-5)

        # grab all of the "optimizable" parameters from the base networks
        self.joint_params = [self.s0, self.obs_logvar, self.step_scales]
        self.joint_params.extend(self.p_zi_given_xi.mlp_params)
        self.joint_params.extend(self.p_sip1_given_zi.mlp_params)
        self.joint_params.extend(self.p_x_given_si.mlp_params)
        self.joint_params.extend(self.q_zi_given_xi.mlp_params)

        #################################
        # CONSTRUCT THE KLD-BASED COSTS #
        #################################
        self.kld_p, self.kld_q, self.kld_g, self.kld_s = self._construct_kld_costs(p=1.0)
        self.kld_costs = (self.lam_kld_p[0] * self.kld_p) + \
                         (self.lam_kld_q[0] * self.kld_q) + \
                         (self.lam_kld_g[0] * self.kld_g) + \
                         (self.lam_kld_s[0] * self.kld_s)
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
        print("Compiling cost computer...")
        self.compute_raw_costs = self._construct_raw_costs()
        print("Compiling training function...")
        self.train_joint = self._construct_train_joint()
        print("Compiling free-energy sampler...")
        self.compute_fe_terms = self._construct_compute_fe_terms()
        print("Compiling sequence sampler...")
        self.sequence_sampler = self._construct_sequence_sampler()
        # make easy access points for some interesting parameters
        #self.gen_inf_weights = self.p_zi_given_xi.shared_layers[0].W
        return

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

    def set_lam_kld(self, lam_kld_p=0.0, lam_kld_q=1.0, lam_kld_g=0.0, lam_kld_s=0.0):
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
        new_lam = zero_ary + lam_kld_s
        self.lam_kld_s.set_value(to_fX(new_lam))
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
        trajectories from this WalkoutModel, for input matrix xo.
        """
        zi_zmuv = self.rng.normal( \
                size=(self.total_steps, xo.shape[0], self.z_dim), \
                avg=0.0, std=1.0, dtype=theano.config.floatX)
        return zi_zmuv

    def _construct_rev_masks(self, xo):
        """
        Compute the sequential revelation masks for the input batch in xo.
        -- We need to construct mask sequences for both p and q.
        """
        if self.use_rev_masks:
            # make batch copies of self.rev_masks_p and self.rev_masks_q
            pmasks = self.rev_masks_p.dimshuffle(0,'x',1).repeat(xo.shape[0], axis=1)
            qmasks = self.rev_masks_q.dimshuffle(0,'x',1).repeat(xo.shape[0], axis=1)
        else:
            pm_list = []
            qm_list = []
            # make a zero mask that does nothing
            zero_mask = T.alloc(0.0, 1, xo.shape[0], xo.shape[1])
            # generate independently sampled masks for each revelation block
            for rb in self.rev_sched:
                # make a random binary mask with ones at rate rb[1]
                rand_vals = self.rng.uniform( \
                        size=(1, xo.shape[0], xo.shape[1]), \
                        low=0.0, high=1.0, dtype=theano.config.floatX)
                rand_mask = rand_vals < rb[1]
                # append the masks for this revleation block to the mask lists
                #
                # the guide policy (in q) gets to peek at the values that will be
                # revealed to the primary policy (in p) for the entire block. The
                # primary policy only gets to see these values at end of the final
                # step of the block. Within a given step, values are revealed to q
                # at the beginning of the step, and to p at the end.
                #
                # e.g. in a revelation block with only a single step, the guide
                # policy sees the values at the beginning of the step, which allows
                # it to guide the step. the primary policy only gets to see the
                # values at the end of the step.
                #
                # i.e. a standard variational auto-encoder is equivalent to a
                # sequential revelation and refinement model with only one
                # revelation block, which has one step and a reveal rate of 1.0.
                #
                for refine_step in range(rb[0]-1):
                    pm_list.append(zero_mask)
                    qm_list.append(rand_mask)
                pm_list.append(rand_mask)
                qm_list.append(rand_mask)
            # concatenate each mask list into a 3-tensor
            pmasks = T.cast(T.concatenate(pm_list, axis=0), 'floatX')
            qmasks = T.cast(T.concatenate(qm_list, axis=0), 'floatX')
        return [pmasks, qmasks]

    def _construct_nll_costs(self, si, xo, nll_mask):
        """
        Construct the negative log-likelihood part of free energy.
        -- only check NLL where nll_mask == 1
        """
        xh = self._from_si_to_x( si )
        if self.x_type == 'bernoulli':
            ll_costs = log_prob_bernoulli(xo, xh, mask=nll_mask)
        else:
            ll_costs = log_prob_gaussian2(xo, xh, \
                    log_vars=self.bounded_logvar, mask=nll_mask)
        nll_costs = -ll_costs.flatten()
        return nll_costs

    def _construct_kld_s(self, s_i, s_j):
        """
        Compute KL(s_i || s_j) -- assuming bernoullish outputs
        """
        x_i = self._from_si_to_x( s_i )
        x_j = self._from_si_to_x( s_j )
        kld_s = (x_i * (T.log(x_i)  - T.log(x_j))) + \
                ((1.0 - x_i) * (T.log(1.0-x_i) - T.log(1.0-x_j)))
        sum_kld = T.sum(kld_s, axis=1)
        return sum_kld

    def _construct_kld_costs(self, p=1.0):
        """
        Construct the policy KL-divergence part of cost to minimize.
        """
        kld_pis = []
        kld_qis = []
        kld_gis = []
        kld_sis = []
        s0 = 0.0*self.si[0] + self.s0
        for i in range(self.total_steps):
            kld_pis.append(T.sum(self.kldi_p2q[i]**p, axis=1))
            kld_qis.append(T.sum(self.kldi_q2p[i]**p, axis=1))
            kld_gis.append(T.sum(self.kldi_p2g[i]**p, axis=1))
            if i == 0:
                kld_sis.append(self._construct_kld_s(self.si[i], s0))
            else:
                kld_sis.append(self._construct_kld_s(self.si[i], self.si[i-1]))
        # compute the batch-wise costs
        kld_pi = sum(kld_pis)
        kld_qi = sum(kld_qis)
        kld_gi = sum(kld_gis)
        kld_si = sum(kld_sis)
        return [kld_pi, kld_qi, kld_gi, kld_si]

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
        pmasks, qmasks = self._construct_rev_masks(xo)
        # construct values to output
        nll = self.nll_costs.flatten()
        kld = self.kld_q.flatten()
        # compile theano function for a one-sample free-energy estimate
        fe_term_sample = theano.function(inputs=[ xo ], \
                outputs=[nll, kld], \
                givens={self.x_out: xo, \
                        self.zi_zmuv: zizmuv, \
                        self.p_masks: pmasks, \
                        self.q_masks: qmasks}, \
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
        pmasks, qmasks = self._construct_rev_masks(xo)
        # compile theano function for computing the costs
        all_step_costs = [self.nlli, self.kldi_q2p, self.kldi_p2q, self.kldi_p2g]
        cost_func = theano.function(inputs=[ xo ], \
                    outputs=all_step_costs, \
                    givens={self.x_out: xo, \
                            self.zi_zmuv: zizmuv, \
                            self.p_masks: pmasks, \
                            self.q_masks: qmasks}, \
                    updates=self.scan_updates, \
                    on_unused_input='ignore')
        # make a function for computing batch-based estimates of costs.
        #   _step_nlls: the expected NLL cost for each step
        #   _step_klds: the expected KL(q||p) cost for each step
        #   _kld_q2p: the expected KL(q||p) cost for each latent dim
        #   _kld_p2q: the expected KL(p||q) cost for each latent dim
        #   _kld_p2g: the expected KL(p||N(0,I)) cost for each latent dim
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
        pmasks, qmasks = self._construct_rev_masks(xo)
        # collect the outputs to return from this function
        outputs = [self.joint_cost, self.nll_bound, self.nll_cost, \
                   self.kld_cost, self.reg_cost, self.obs_costs]
        # compile the theano function
        func = theano.function(inputs=[ xo ], \
                outputs=outputs, \
                givens={self.x_out: xo, \
                        self.zi_zmuv: zizmuv, \
                        self.p_masks: pmasks, \
                        self.q_masks: qmasks}, \
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
        pmasks, qmasks = self._construct_rev_masks(xo)
        # collect the outputs to return from this function
        states = [self._from_si_to_x(self.s0_full)] + \
                 [self._from_si_to_x(self.si[i]) for i in range(self.total_steps)]
        masks = [self.m0_full] + [self.mi_p[i] for i in range(self.total_steps)]
        outputs = states + masks
        # compile the theano function
        func = theano.function(inputs=[ xo ], \
                outputs=outputs, \
                givens={self.x_out: xo, \
                        self.zi_zmuv: zizmuv, \
                        self.p_masks: pmasks, \
                        self.q_masks: qmasks}, \
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

def load_WalkoutModel_from_file(f_name=None, rng=None):
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
    # now, create a new WalkoutModel based on the loaded data
    xo = T.matrix()
    clone_net = WalkoutModel(rng=rng, \
                         x_out=xo, \
                         p_zi_given_xi=p_zi_given_xi, \
                         p_sip1_given_zi=p_sip1_given_zi, \
                         p_x_given_si=p_x_given_si, \
                         q_zi_given_xi=q_zi_given_xi, \
                         params=self_dot_params, \
                         shared_param_dicts=self_dot_shared_param_dicts)
    # helpful output
    print("==================================================")
    print("LOADED WalkoutModel WITH PARAMS:")
    for k in self_dot_params:
        print("    {0:s}: {1:s}".format(str(k), str(self_dot_params[k])))
    print("==================================================")
    return clone_net



if __name__=="__main__":
    print("Hello world!")







##############
# EYE BUFFER #
##############
