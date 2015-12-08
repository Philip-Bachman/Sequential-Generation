##################################################################
# Code for networks and whatnot to use in variationalish stuff.  #
##################################################################

# basic python
import numpy as np
import numpy.random as npr
from collections import OrderedDict
import cPickle

# theano business
import theano
import theano.tensor as T
#from theano.tensor.shared_randomstreams import RandomStreams as RandStream
from theano.sandbox.cuda.rng_curand import CURAND_RandomStreams as RandStream

# phil's sweetness
from NetLayers import HiddenLayer, relu_actfun, softplus_actfun
from HelperFuncs import constFX, to_fX

######################################################
# MULTI-PURPOSE, MULTI-HEADED NETWORK IMPLEMENTATION #
######################################################

class HydraNet(object):
    """
    A net that turns one input into one or more outputs.

    Some shared hidden layers feed into one or more output layers.

    Parameters:
        rng: a numpy.random RandomState object
        Xd: symbolic input matrix for inputs
        params: a dict of parameters describing the desired network:
            shared_config: list of "layer descriptions" for the shared layers
            output_config: list of "layer descriptions" for the output layers
        shared_param_dicts: parameters for this HydraNet
    """
    def __init__(self,
            rng=None,
            Xd=None,
            params=None,
            shared_param_dicts=None):
        # Setup a shared random generator for this network
        self.rng = RandStream(rng.randint(1000000))
        # Grab the symbolic input matrix
        self.Xd = Xd
        #####################################################
        # Process user-supplied parameters for this network #
        #####################################################
        self.params = params
        if 'build_theano_funcs' in params:
            self.build_theano_funcs = params['build_theano_funcs']
        else:
            self.build_theano_funcs = True
        if 'init_scale' in params:
            self.init_scale = params['init_scale']
        else:
            self.init_scale = 1.0
        # Check if the params for this net were given a priori. This option
        # will be used for creating "clones" of an inference network, with all
        # of the network parameters shared between clones.
        if shared_param_dicts is None:
            # This is not a clone, and we will need to make a dict for
            # referring to the parameters of each network layer
            self.shared_param_dicts = {'shared': [], 'output': []}
            self.is_clone = False
        else:
            # This is a clone, and its layer parameters can be found by
            # referring to the given param dict (i.e. shared_param_dicts).
            self.shared_param_dicts = shared_param_dicts
            self.is_clone = True
        # Get the configuration/prototype for this network.
        self.shared_config = params['shared_config']
        self.output_config = params['output_config']

        ###
        self.shared_layers = []
        #########################################
        # Initialize the shared part of network #
        #########################################
        for sl_num, sl_desc in enumerate(self.shared_config):
            l_name = "shared_layer_{0:d}".format(sl_num)
            if not self.is_clone:
                ##########################################
                # Initialize a layer with new parameters #
                ##########################################
                new_layer = HiddenLayer(rng=rng,
                        layer_description=sl_desc,
                        name=l_name, W_scale=self.init_scale)
                self.shared_layers.append(new_layer)
                self.shared_param_dicts['shared'].append(
                        new_layer.shared_param_dicts)
            else:
                ##################################################
                # Initialize a layer with some shared parameters #
                ##################################################
                init_params = self.shared_param_dicts['shared'][sl_num]
                new_layer = HiddenLayer(rng=rng,
                        layer_description=sl_desc,
                        W=init_params['W'], b=init_params['b'],
                        b_in=init_params['b_in'], s_in=init_params['s_in'],
                        name=l_name, W_scale=self.init_scale)
                self.shared_layers.append(new_layer)
        ################################
        # Initialize the output layers #
        ################################
        self.output_layers = []
        # take input from the output of the shared network
        for ol_num, ol_desc in enumerate(self.output_config):
            ol_name = "output_layer_{0:d}".format(ol_num)
            if not self.is_clone:
                ##########################################
                # Initialize a layer with new parameters #
                ##########################################
                new_layer = HiddenLayer(rng=rng,
                        layer_description=ol_desc,
                        name=ol_name, W_scale=self.init_scale)
                self.output_layers.append(new_layer)
                self.shared_param_dicts['output'].append(
                        new_layer.shared_param_dicts)
            else:
                ##################################################
                # Initialize a layer with some shared parameters #
                ##################################################
                init_params = self.shared_param_dicts['output'][ol_num]
                new_layer = HiddenLayer(rng=rng,
                        layer_description=ol_desc,
                        W=init_params['W'], b=init_params['b'],
                        b_in=init_params['b_in'], s_in=init_params['s_in'],
                        name=ol_name, W_scale=self.init_scale)
                self.output_layers.append(new_layer)

        # mash all the parameters together, into a list.
        self.mlp_params = []
        for layer in self.shared_layers:
            self.mlp_params.extend(layer.params)
        for layer in self.output_layers:
            self.mlp_params.extend(layer.params)
        return

    def apply(self, X, use_drop=False):
        """
        Pass input X through this HydraNet and get the resulting outputs.
        """
        # pass activations through the shared layers
        shared_acts = [X]
        for layer in self.shared_layers:
            layer_acts, _ = layer.apply(shared_acts[-1], use_drop=use_drop)
            shared_acts.append(layer_acts)
        shared_output = shared_acts[-1]
        # compute outputs of the output layers
        outputs = []
        for layer in self.output_layers:
            _, layer_acts = layer.apply(shared_output, use_drop=use_drop)
            outputs.append(layer_acts)
        return outputs

    def apply_shared(self, X, use_drop=False):
        """
        Pass input X through this HydraNet's shared layers.
        """
        # pass activations through the shared layers
        shared_acts = [X]
        for layer in self.shared_layers:
            layer_acts, _ = layer.apply(shared_acts[-1], use_drop=use_drop)
            shared_acts.append(layer_acts)
        shared_output = shared_acts[-1]
        return shared_output

    def init_biases(self, b_init=0.0, b_std=1e-2):
        """
        Initialize the biases in all shred layers to some constant.
        """
        for layer in self.shared_layers:
            b_vec = (0.0 * layer.b.get_value(borrow=False)) + b_init
            b_vec = b_vec + (b_std * npr.randn(*b_vec.shape))
            layer.b.set_value(to_fX(b_vec))
        return

    def shared_param_clone(self, rng=None, Xd=None):
        """
        Return a clone of this network, with shared parameters but with
        different symbolic input variables.
        """
        clone_net = HydraNet(rng=rng, Xd=Xd, params=self.params, \
                shared_param_dicts=self.shared_param_dicts)
        return clone_net

    def forked_param_clone(self, rng=None, Xd=None):
        """
        Return a clone of this network, with forked copies of the current
        shared parameters of this HydraNet, with different symbolic inputs.
        """
        new_spds = {}
        old_spds = self.shared_param_dicts
        # shared param dicts is nested like: dict of list of dicts
        # i.e., spd[k] is a list and spd[k][i] is a dict
        for k1 in old_spds:
            new_spds[k1] = []
            for i in range(len(old_spds[k1])):
                new_spds[k1].append({})
                for k2 in old_spds[k1][i]:
                    old_sp = old_spds[k1][i][k2]
                    old_sp_forked = old_sp.get_value(borrow=False)
                    new_sp = theano.shared(value=old_sp_forked)
                    new_spds[k1][i][k2] = new_sp
        clone_net = HydraNet(rng=rng, Xd=Xd, params=self.params, \
                shared_param_dicts=new_spds)
        return clone_net

    def save_to_file(self, f_name=None):
        """
        Dump important stuff to a Python pickle, so that we can reload this
        model later. We'll pickle everything required to create a clone of
        this model given the pickle and the rng/Xd params to the cloning
        function: "HydraNet.shared_param_clone()".
        """
        assert(not (f_name is None))
        f_handle = file(f_name, 'wb')
        # dump the dict self.params, which just holds "simple" python values
        cPickle.dump(self.params, f_handle, protocol=-1)
        # make a copy of self.shared_param_dicts, with numpy arrays in place
        # of the theano shared variables
        numpy_param_dicts = {'shared': [], 'output': []}
        for layer_group in ['shared', 'output']:
            for shared_dict in self.shared_param_dicts[layer_group]:
                numpy_dict = {}
                for key in shared_dict:
                    numpy_dict[key] = shared_dict[key].get_value(borrow=False)
                numpy_param_dicts[layer_group].append(numpy_dict)
        # dump the numpy version of self.shared_param_dicts to pickle file
        cPickle.dump(numpy_param_dicts, f_handle, protocol=-1)
        f_handle.close()
        return

    def save_to_dict(self):
        """
        Dump important stuff to a dict capable of rebooting the model.
        """
        model_dict = {}
        # dump the dict self.params, which just holds "simple" python values
        model_dict['params'] = self.params
        # make a copy of self.shared_param_dicts, with numpy arrays in place
        # of the theano shared variables
        numpy_param_dicts = {'shared': [], 'output': []}
        for layer_group in ['shared', 'output']:
            for shared_dict in self.shared_param_dicts[layer_group]:
                numpy_dict = {}
                for key in shared_dict:
                    numpy_dict[key] = shared_dict[key].get_value(borrow=False)
                numpy_param_dicts[layer_group].append(numpy_dict)
        # dump the numpy version of self.shared_param_dicts to the dict
        model_dict['numpy_param_dicts'] = numpy_param_dicts
        return model_dict

def load_hydranet_from_file(f_name=None, rng=None, Xd=None, \
                            new_params=None):
    """
    Load a clone of some previously trained model.
    """
    assert(not (f_name is None))
    pickle_file = open(f_name)
    # load basic parameters
    self_dot_params = cPickle.load(pickle_file)
    if not (new_params is None):
        for k in new_params:
            self_dot_params[k] = new_params[k]
    # load numpy arrays that will be converted to Theano shared arrays
    self_dot_numpy_param_dicts = cPickle.load(pickle_file)
    self_dot_shared_param_dicts = {'shared': [], 'output': []}
    for layer_group in ['shared', 'output']:
        # go over the list of parameter dicts in this layer group
        for numpy_dict in self_dot_numpy_param_dicts[layer_group]:
            shared_dict = {}
            for key in numpy_dict:
                # convert each numpy array to a Theano shared array
                val = to_fX(numpy_dict[key])
                shared_dict[key] = theano.shared(val)
            self_dot_shared_param_dicts[layer_group].append(shared_dict)
    # now, create a HydraNet with the configuration we just unpickled
    clone_net = HydraNet(rng=rng, Xd=Xd, params=self_dot_params, \
                         shared_param_dicts=self_dot_shared_param_dicts)
    # helpful output
    print("==================================================")
    print("LOADED HydraNet WITH PARAMS:")
    for k in self_dot_params:
        print("    {0:s}: {1:s}".format(str(k), str(self_dot_params[k])))
    print("==================================================")
    return clone_net

def load_hydranet_from_dict(model_dict, rng=None, Xd=None, \
                            new_params=None):
    """
    Load a clone of some previously trained model.
    """
    # load basic parameters
    self_dot_params = model_dict['params']
    if not (new_params is None):
        for k in new_params:
            self_dot_params[k] = new_params[k]
    # load numpy arrays that will be converted to Theano shared arrays
    self_dot_numpy_param_dicts = model_dict['numpy_param_dicts']
    self_dot_shared_param_dicts = {'shared': [], 'output': []}
    for layer_group in ['shared', 'output']:
        # go over the list of parameter dicts in this layer group
        for numpy_dict in self_dot_numpy_param_dicts[layer_group]:
            shared_dict = {}
            for key in numpy_dict:
                # convert each numpy array to a Theano shared array
                val = to_fX(numpy_dict[key])
                shared_dict[key] = theano.shared(val)
            self_dot_shared_param_dicts[layer_group].append(shared_dict)
    # now, create a HydraNet with the configuration we just unpacked
    clone_net = HydraNet(rng=rng, Xd=Xd, params=self_dot_params, \
                         shared_param_dicts=self_dot_shared_param_dicts)
    # helpful output
    print("==================================================")
    print("LOADED HydraNet WITH PARAMS:")
    for k in self_dot_params:
        print("    {0:s}: {1:s}".format(str(k), str(self_dot_params[k])))
    print("==================================================")
    return clone_net







if __name__=="__main__":
    # Derp
    print("NO TEST/DEMO CODE FOR NOW.")
