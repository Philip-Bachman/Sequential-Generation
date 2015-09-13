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
from blocks.roles import add_role, WEIGHT, BIAS, PARAMETER, AUXILIARY

from BlocksAttention import ZoomableAttention2d
from RAMBlocks import SimpleAttentionReader2d

from PIL import Image
import pylab


#--------------------------------------------------------------------
N = 16
height = 480
width =  640
con_dim = 100

# symbolic variables for theano function building
I_ = T.matrix()
W_ = T.matrix()
center_y_ = T.vector()
center_x_ = T.vector()
delta_ = T.vector()
gamma1_ = T.vector()
gamma2_ = T.vector()

SAR2d = SimpleAttentionReader2d(x_dim=height*width, con_dim=con_dim, \
                                height=height, width=width, N=N, \
                                init_scale=2.0)

READ_ = SAR2d.direct_read(x=I_, center_y=center_y_, center_x=center_x_, \
                          delta=delta_, gamma1=gamma1_, gamma2=gamma2_)
WRITE_ = SAR2d.direct_write(windows=W_, center_y=center_y_, \
                            center_x=center_x_, delta=delta_)
M12_ = SAR2d.direct_att_map(center_y=center_y_, center_x=center_x_, \
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
I = I.resize((width, height)).convert('L')

I = np.asarray(I).reshape( (width*height) )
I = I / 255.

center_y = 0.0
center_x = 0.0
delta = 1.
sigma = 1.
gamma = 1.

def vectorize(*args):
    return [a.reshape((1,)+a.shape) for a in args]

I, center_y, center_x, delta, sigma, gamma = \
    vectorize(I, np.array(center_y), np.array(center_x), np.array(delta), \
              np.array(sigma), np.array(gamma))

#
I = I.repeat(3, axis=0)
center_y = center_y.repeat(3, axis=0)
center_x = center_x.repeat(3, axis=0)
delta = delta.repeat(3, axis=0)
sigma = sigma.repeat(3, axis=0)
gamma = gamma.repeat(3, axis=0)

print("I.shape: {}".format(I.shape))
print("center_y.shape: {}".format(center_y.shape))
print("center_x.shape: {}".format(center_x.shape))
print("delta.shape: {}".format(delta.shape))
print("sigma.shape: {}".format(sigma.shape))
print("gamma.shape: {}".format(gamma.shape))

# apply the foveated attention mechanism
R12 = read_sar2d(I, center_y, center_x, delta, gamma, 4.0*gamma)
I12 = write_sar2d(R12, center_y, center_x, delta)
M12 = att_map_sar2d(center_y, center_x, delta, gamma, 4.0*gamma)
R12 = R12[[0],:]
I12 = I12[[0],:]
M12 = M12[[0],:]

# split results into their inner/outer parts
R1 = R12[:,:(SAR2d.read_dim/2)]
R2 = R12[:,(SAR2d.read_dim/2):]
I1 = I12[:,:(width*height)]
I2 = I12[:,(width*height):]
I12 = I1 + I2
M1 = M12[:,:(width*height)]
M2 = M12[:,(width*height):]
M12 = M1 + M2

# visualize results
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
pylab.imshow(I1.reshape([height, width]), interpolation='nearest')
pylab.show()
pylab.savefig('AAA_I1.png')

pylab.figure()
pylab.gray()
pylab.imshow(I2.reshape([height, width]), interpolation='nearest')
pylab.show()
pylab.savefig('AAA_I2.png')

pylab.figure()
pylab.gray()
pylab.imshow(I12.reshape([height, width]), interpolation='nearest')
pylab.show()
pylab.savefig('AAA_I12.png')

pylab.figure()
pylab.gray()
pylab.imshow(M1.reshape([height, width]), interpolation='nearest')
pylab.show()
pylab.savefig('AAA_M1.png')

pylab.figure()
pylab.gray()
pylab.imshow(M2.reshape([height, width]), interpolation='nearest')
pylab.show()
pylab.savefig('AAA_M2.png')

pylab.figure()
pylab.gray()
pylab.imshow(M12.reshape([height, width]), interpolation='nearest')
pylab.show()
pylab.savefig('AAA_M12.png')
