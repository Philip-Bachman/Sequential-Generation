##################################################################
# Code for testing the variational Multi-Stage Generative Model. #
##################################################################

from __future__ import print_function, division

# basic python
import cPickle as pickle
import cPickle
from PIL import Image
import numpy as np
import numpy.random as npr
from collections import OrderedDict

# theano business
import theano
import theano.tensor as T

# phil's sweetness
import utils

def process_samples(step_type='add', data_name='MNIST'):
    # sample several interchangeable versions of the model
    if data_name == 'MNIST':
        conditions = [{'occ_dim': 0, 'drop_prob': 0.8}, \
                      {'occ_dim': 16, 'drop_prob': 0.0}]
    if data_name == 'SVHN':
        conditions = [{'occ_dim': 0, 'drop_prob': 0.8}, \
                      {'occ_dim': 17, 'drop_prob': 0.0}]
    if data_name == 'TFD':
        conditions = [{'occ_dim': 0, 'drop_prob': 0.8}, \
                      {'occ_dim': 25, 'drop_prob': 0.0}]
    for cond_dict in conditions:
        occ_dim = cond_dict['occ_dim']
        drop_prob = cond_dict['drop_prob']
        dp_int = int(100.0 * drop_prob)

        # save the samples to a pkl file, in their numpy array form
        sample_pkl_name = "IMP-{}-OD{}-DP{}-{}.pkl".format(data_name, occ_dim, dp_int, step_type)
        pickle_file = open(sample_pkl_name)
        samples = cPickle.load(pickle_file)
        pickle_file.close()
        print("Loaded some samples from: {}".format(sample_pkl_name))

        sample_list = []
        for i in range(samples.shape[0]):
            sample_list.append(samples[i,:,:])
        # downsample the sequence....
        #keep_idx = range(len(sample_list))
        keep_idx = [0, 2, 4, 6, 9, 12, 15]
        sample_list = [sample_list[i] for i in keep_idx]

        seq_len = len(sample_list)
        samp_count = sample_list[0].shape[0]
        obs_dim = sample_list[0].shape[1]

        seq_samps = np.zeros((seq_len*samp_count, obs_dim))
        idx = 0
        for s1 in range(samp_count):
            for s2 in range(seq_len):
                seq_samps[idx] = sample_list[s2][s1,:].ravel()
                idx += 1
        sample_img_name = "IMP-{}-OD{}-DP{}-{}.png".format(data_name, occ_dim, dp_int, step_type)

        row_count = int(samp_count / 16)
        print("row_count: {}".format(row_count))
        utils.visualize_samples(seq_samps, sample_img_name, num_rows=row_count)
    return


if __name__=="__main__":
    #process_samples(step_type='add', data_name='MNIST')
    #process_samples(step_type='jump', data_name='MNIST')
    process_samples(step_type='add', data_name='TFD')
    process_samples(step_type='add', data_name='SVHN')
