from __future__ import division, print_function

import logging
import theano
import numpy
import cPickle

import theano
from theano import tensor
from theano import tensor as T
import numpy as np
from collections import OrderedDict

from blocks.graph import ComputationGraph
from blocks.filter import VariableFilter
from blocks.bricks.base import application, _Brick, Brick, lazy
from blocks.bricks.recurrent import BaseRecurrent, recurrent
from blocks.initialization import Constant, IsotropicGaussian, Orthogonal
from blocks.bricks import Random, MLP, Linear, Tanh, Softmax, Sigmoid, Initializable
from blocks.bricks import Tanh, Identity, Activation, Feedforward
from blocks.bricks.cost import BinaryCrossEntropy
from blocks.utils import shared_floatx_nans
from blocks.roles import add_role, WEIGHT, BIAS, PARAMETER, AUXILIARY

from BlocksAttention import ZoomableAttentionWindow
from SRRBlocks import SimpleAttentionReader2d
from DKCode import get_adam_updates
from HelperFuncs import constFX, to_fX
from LogPDFs import log_prob_bernoulli, gaussian_kld

from PIL import Image


N = 40
height = 480
width =  640

#------------------------------------------------------------------------
att = ZoomableAttentionWindow(height, width, N)

I_ = T.matrix()
center_y_ = T.vector()
center_x_ = T.vector()
delta_ = T.vector()
sigma_ = T.vector()
W_ = att.read(I_, center_y_, center_x_, delta_, sigma_)

do_read = theano.function(inputs=[I_, center_y_, center_x_, delta_, sigma_],
                          outputs=W_, allow_input_downcast=True)

W_ = T.matrix()
center_y_ = T.vector()
center_x_ = T.vector()
delta_ = T.vector()
sigma_ = T.vector()
gamma_ = T.vector()
I_ = att.write(W_, center_y_, center_x_, delta_, sigma_)

do_write = theano.function(inputs=[W_, center_y_, center_x_, delta_, sigma_],
                           outputs=I_, allow_input_downcast=True)


I_ = T.matrix()
W_ = T.matrix()
center_y_ = T.vector()
center_x_ = T.vector()
delta_ = T.vector()
gamma1_ = T.vector()
gamma2_ = T.vector()

SAR2d = SimpleAttentionReader2d(x_dim=height*width, con_dim=100, \
                                height=height, width=width, N=N)

READ_ = SAR2d.direct_read(im=I_, center_y=center_y_, center_x=center_x_, \
                          delta=delta_, gamma1=gamma1_, gamma2=gamma2_)
WRITE_ = SAR2d.direct_write(windows=W_, center_y=center_y_, \
                            center_x=center_x_, delta=delta_)
M12_, _, _ = SAR2d.direct_att_map(center_y=center_y_, center_x=center_x_, \
                                  delta=delta_, gamma1=gamma1_, gamma2=gamma2_)

read_sar2d = theano.function(inputs=[I_, center_y_, center_x_, delta_, gamma1_, gamma2_], \
                             outputs=READ_, allow_input_downcast=True, on_unused_input='ignore')

write_sar2d = theano.function(inputs=[W_, center_y_, center_x_, delta_], \
                              outputs=WRITE_, allow_input_downcast=True, on_unused_input='ignore')

att_map_sar2d = theano.function(inputs=[center_y_, center_x_, delta_, gamma1_, gamma2_], \
                                outputs=M12_, allow_input_downcast=True, on_unused_input='ignore')

cg = ComputationGraph([READ_])
joint_params = VariableFilter(roles=[PARAMETER])(cg.variables)
print("joint_params:")
print("{}".format(str(joint_params)))

#------------------------------------------------------------------------

I = Image.open("Vau08.jpg")
I = I.resize((640, 480)).convert('L')

I = np.asarray(I).reshape( (width*height) )
I = I / 255.

center_y = 200.5
center_x = 330.5
delta = 5.
sigma = 2.
gamma = 1.

def vectorize(*args):
    return [a.reshape((1,)+a.shape) for a in args]

I, center_y, center_x, delta, sigma, gamma = \
    vectorize(I, np.array(center_y), np.array(center_x), np.array(delta), np.array(sigma), np.array(gamma))

#
I = I.repeat(3, axis=0)
center_y = center_y.repeat(3, axis=0)
center_x = center_x.repeat(3, axis=0)
delta = delta.repeat(3, axis=0)
sigma = sigma.repeat(3, axis=0)
gamma = gamma.repeat(3, axis=0)

#import ipdb; ipdb.set_trace()

# read from the basic attention zoomer thingy
R0  = do_read(I, center_y, center_x, delta, sigma)
I0 = do_write(R0, center_y, center_x, delta, sigma)
R0 = R0[0]
I0 = I0[0]

# read from the foveated attention mechanism
print("I.shape: {}".format(I.shape))
print("center_y.shape: {}".format(center_y.shape))
print("center_x.shape: {}".format(center_x.shape))
print("delta.shape: {}".format(delta.shape))
print("sigma.shape: {}".format(sigma.shape))
print("gamma.shape: {}".format(gamma.shape))
R12 = read_sar2d(I, center_y, center_x, delta, gamma, 4.0*gamma)
I12 = write_sar2d(R12, center_y, center_x, delta)
print("I12.shape: {}".format(I12.shape))
M12 = att_map_sar2d(center_y, center_x, delta, gamma, 4.0*gamma)
print("M12.shape: {}".format(M12.shape))
R12 = R12[[0],:]
I12 = I12[[0],:]
M12 = M12[[0],:]

R1 = R12[:,:(SAR2d.read_dim/2)]
R2 = R12[:,(SAR2d.read_dim/2):]
print("R1.shape: {}".format(R1.shape))
print("R2.shape: {}".format(R2.shape))
I1 = I12[:,:(width*height)]
I2 = I12[:,(width*height):]
I12 = I1 + I2

M1 = M12[:,:(width*height)]
M2 = M12[:,(width*height):]
print("M1.shape: {}".format(M1.shape))
print("M2.shape: {}".format(M2.shape))
M12 = M1 + M2


import pylab
pylab.figure()
pylab.gray()
pylab.imshow(I[0].reshape([height, width]), interpolation='nearest')
pylab.savefig('AAA_SRC.png')

pylab.figure()
pylab.gray()
pylab.imshow(R1.reshape([N, N]), interpolation='nearest')
pylab.savefig('AAA_R1.png')

pylab.figure()
pylab.gray()
pylab.imshow(R2.reshape([N, N]), interpolation='nearest')
pylab.savefig('AAA_R2.png')

pylab.figure()
pylab.gray()
pylab.imshow(I0.reshape([height, width]), interpolation='nearest')
pylab.show() #block=True)
pylab.savefig('AAA_I0.png')

pylab.figure()
pylab.gray()
pylab.imshow(I1.reshape([height, width]), interpolation='nearest')
pylab.show() #block=True)
pylab.savefig('AAA_I1.png')

pylab.figure()
pylab.gray()
pylab.imshow(I2.reshape([height, width]), interpolation='nearest')
pylab.show() #block=True)
pylab.savefig('AAA_I2.png')

pylab.figure()
pylab.gray()
pylab.imshow(I12.reshape([height, width]), interpolation='nearest')
pylab.show() #block=True)
pylab.savefig('AAA_I12.png')

pylab.figure()
pylab.gray()
pylab.imshow(M1.reshape([height, width]), interpolation='nearest')
pylab.show() #block=True)
pylab.savefig('AAA_M1.png')

pylab.figure()
pylab.gray()
pylab.imshow(M2.reshape([height, width]), interpolation='nearest')
pylab.show() #block=True)
pylab.savefig('AAA_M2.png')

pylab.figure()
pylab.gray()
pylab.imshow(M12.reshape([height, width]), interpolation='nearest')
pylab.show() #block=True)
pylab.savefig('AAA_M12.png')

#import ipdb; ipdb.set_trace()
