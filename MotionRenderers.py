#!/ysr/bin/env python

from __future__ import division


import utils as utils
import numpy as np
import numpy.random as npr
import theano
import theano.tensor as T
import time


def my_batched_dot(A, B):
    """
    Compute: [np.dot(A[i,:,:], B[i,:,:]) for i in range(A.shape[0])]
    """
    C = A.dimshuffle([0,1,2,'x']) * B.dimshuffle([0,'x',1,2])
    return C.sum(axis=-2)

############################################
############################################
## Class for painting objects into images ##
############################################
############################################

class ObjectPainter(object):
    def __init__(self, img_height, img_width, obj_type='circle', obj_scale=0.2):
        """
        A class for drawing a few simple objects with subpixel resolution.
        """
        self.img_height = img_height
        self.img_width = img_width
        self.obj_type = obj_type
        self.obj_scale = obj_scale
        # make coordinate system for points in the object to render
        obj_x_coords, obj_y_coords = self._construct_obj_coords( \
                    obj_type=self.obj_type, obj_scale=self.obj_scale)
        self.obj_x = T.constant(obj_x_coords)
        self.obj_y = T.constant(obj_y_coords)
        self.obj_x_range = [np.min(obj_x_coords), np.max(obj_x_coords)]
        self.obj_y_range = [np.min(obj_y_coords), np.max(obj_y_coords)]
        # make coordinate system for x and y location in the image.
        #   -- image coordinates for the smallest dimension range over
        #      [-init_scale....init_scale], and coordinates for the largest
        #      dimension are at the same scale, but over a larger range.
        img_x_coords, img_y_coords = self._construct_img_coords( \
                    x_dim=self.img_width, y_dim=self.img_height)
        self.img_x = T.constant(img_x_coords)
        self.img_y = T.constant(img_y_coords)
        self.img_x_range = [np.min(img_x_coords), np.max(img_x_coords)]
        self.img_y_range = [np.min(img_y_coords), np.max(img_y_coords)]
        return

    def _construct_obj_coords(self, obj_type='circle', obj_scale=0.2):
        """
        Construct coordinates for circle, square, or cross.
        """
        if obj_type == 'circle':
            coords = [(-1, 3), (0, 3), (1, 3), (2, 2), \
                      (3, 1), (3, 0), (3, -1), (2, -2), \
                      (1, -3), (0, -3), (-1, -3), (-2, -2), \
                      (-3, -1), (-3, 0), (-3, 1), (-2, 2)]
        elif obj_type == 'square':
            coords = [(-2, 2), (-1, 2), (0, 2), (1, 2), \
                      (2, 2), (2, 1), (2, 0), (2, -1), \
                      (2, -2), (1, -2), (0, -2), (-1, -2), \
                      (-2, -2), (-2, -1), (-2, 0), (-2, 1)]
        elif obj_type == 'cross':
            coords = [(0, 3), (0, 2), (0, 1), (0, 0), \
                      (1, 0), (2, 0), (3, 0), (0, -1), \
                      (0, -2), (0, -3), (-1, 0), (-2, 0), \
                      (-3, 0)]
        else:
            coords = [(-1, 1), (0, 1), (1, 1), (1, 0), \
                      (1, -1), (0, -1), (-1, -1), (-1, 0)]
        x_coords = np.asarray([float(pt[0]) for pt in coords])
        y_coords = np.asarray([float(pt[1]) for pt in coords])
        rescale = max(np.max(x_coords), np.max(y_coords))
        x_coords = (obj_scale / rescale) * x_coords
        y_coords = (obj_scale / rescale) * y_coords
        x_coords = x_coords.astype(theano.config.floatX)
        y_coords = y_coords.astype(theano.config.floatX)
        return x_coords, y_coords

    def _construct_img_coords(self, x_dim=32, y_dim=32):
        """
        Construct coordinates for all points in the base images.
        """
        min_dim = float( min(x_dim, y_dim) )
        x_scale = x_dim / min_dim
        y_scale = y_dim / min_dim
        xc = x_scale * np.linspace(start=-1., stop=1., num=x_dim)
        yc = y_scale * np.linspace(start=-1., stop=1., num=y_dim)
        coords = []
        for x_idx in range(x_dim):
            for y_idx in range(y_dim):
                coords.append((xc[x_idx], yc[y_idx]))
        x_coords = np.asarray([float(pt[0]) for pt in coords])
        y_coords = np.asarray([float(pt[1]) for pt in coords])
        x_coords = x_coords.astype(theano.config.floatX)
        y_coords = y_coords.astype(theano.config.floatX)
        return x_coords, y_coords

    def filterbank_matrices(self, center_y, center_x, delta, sigma):
        """
        Create a Fy and a Fx

        Parameters
        ----------
        center_y : T.vector (shape: batch_size)
        center_x : T.vector (shape: batch_size)
            Y and X center coordinates for the attention window
        delta : T.vector (shape: batch_size)
        sigma : T.vector (shape: batch_size)

        Returns
        -------
            FY, FX
        """
        tol = 1e-4
        # construct x and y coordinates for the grid points
        obj_x = center_x.dimshuffle(0, 'x') + \
                (delta.dimshuffle(0, 'x') * self.obj_x)
        obj_y = center_y.dimshuffle(0, 'x') + \
                (delta.dimshuffle(0, 'x') * self.obj_y)

        # construct unnormalized attention weights for each grid point
        FX = T.exp( -(self.img_x - obj_x.dimshuffle(0,1,'x'))**2. / \
                   (2. * sigma.dimshuffle(0,'x','x')**2.) )
        FY = T.exp( -(self.img_y - obj_y.dimshuffle([0,1,'x']))**2. / \
                   (2. * sigma.dimshuffle(0,'x','x')**2.) )

        # normalize the attention weights
        #FX = FX / (FX.sum(axis=-1).dimshuffle(0, 1, 'x') + tol)
        #FY = FY / (FY.sum(axis=-1).dimshuffle(0, 1, 'x') + tol)
        FX = FX / (T.max(FX.sum(axis=-1)) + tol)
        FY = FY / (T.max(FY.sum(axis=-1)) + tol)
        return FY, FX

    def write(self, center_y, center_x, delta, sigma):
        """
        Write a batch of objects into full sized images.

        Parameters
        ----------
        center_y : :class:`~tensor.TensorVariable`
            Center coordinates for the objects.
            Expected shape: (batch_size,)
        center_x : :class:`~tensor.TensorVariable`
            Center coordinates for the objects.
            Expected shape: (batch_size,)
        delta : :class:`~tensor.TensorVariable`
            Scale for the objects.
            Expected shape: (batch_size,)
        sigma : :class:`~tensor.TensorVariable`
            Std. dev. for Gaussian writing kernel.
            Expected shape: (batch_size,)

        Returns
        -------
        images : :class:`~tensor.TensorVariable`
            images of objects: (batch_size x img_height*img_width)
        """
        # Get separable filterbank
        FY, FX = self.filterbank_matrices(center_y, center_x, delta, sigma)

        # apply...
        FI = FX * FY
        I_raw = T.sum(FI, axis=1)
        I = I_raw / T.max(I_raw)
        return I

####################################################################
####################################################################
## Class for generating random trajectories within a bounding box ##
####################################################################
####################################################################

class TrajectoryGenerator(object):
    def __init__(self, x_range=[-1.,1.], y_range=[-1.,1.], max_speed=0.1):
        """
        A class for generating trajectories in box with given x/y range.
        """
        self.x_range = x_range
        self.y_range = y_range
        self.x_min, self.x_max = x_range
        self.y_min, self.y_max = y_range
        return


if __name__ == "__main__":
    samp_count = 200
    center_x = np.linspace(start=-0.8, stop=0.8, num=samp_count).astype(theano.config.floatX)
    center_y = np.linspace(start=-0.8, stop=0.8, num=samp_count).astype(theano.config.floatX)
    delta = np.ones((samp_count,)).astype(theano.config.floatX)
    sigma = np.ones((samp_count,)).astype(theano.config.floatX)

    OPTR = ObjectPainter(32, 32, obj_type='cross', obj_scale=0.2)

    _center_x = T.vector()
    _center_y = T.vector()
    _delta = T.vector()
    _sigma = T.vector()

    _W = OPTR.write(_center_y, _center_x, _delta, _sigma)
    write_func = theano.function(inputs=[_center_y, _center_x, _delta, _sigma], \
                                 outputs=_W)

    start_time = time.time()
    # test the writer function
    gen_count = 100
    for i in range(gen_count):
        W = write_func(center_y, center_x, delta, 0.05*sigma)
    end_time = time.time()
    render_time = end_time - start_time
    render_frames = gen_count * samp_count
    render_fps = render_frames / render_time
    print("RENDER FPS: {0:.2f}".format(render_fps))

    utils.visualize_samples(W, "AAAAA.png", num_rows=10)
