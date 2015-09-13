#!/ysr/bin/env python

from __future__ import division

import numpy as np
import numpy.random as npr

import theano
import theano.tensor as T

from theano import tensor

#-----------------------------------------------------------------------------

def my_batched_dot(A, B):
    """
    Batched version of dot-product.

    For A[dim_1, dim_2, dim_3] and B[dim_1, dim_3, dim_4] this
    is \approx equal to:

    for i in range(dim_1):
        C[i,:,:] = T.dot(A[i,:,:], B[i,:,:])

    Returns
    -------
        C : shape (dim_1 \times dim_2 \times dim_4)
    """
    C = A.dimshuffle([0,1,2,'x']) * B.dimshuffle([0,'x',1,2])
    return C.sum(axis=-2)

#-----------------------------------------------------------------------------

class ZoomableAttention2d(object):
    def __init__(self, img_height, img_width, N, init_scale=2.0):
        """
        A zoomable attention window for images.

        Parameters
        ----------
        img_height, img_width : int
            shape of the images
        N :
            $N \times N$ attention window size
        init_scale :
            initial scaling for source images vs. attention grid
        """
        self.img_height = img_height
        self.img_width = img_width
        self.N = N
        self.init_scale = init_scale
        # make offsets for internal dispersement of grid points.
        #   -- internal grid coordinates range over [-1...+1]
        offsets = np.arange(N) - (N / 2.0) + 0.5
        offsets = offsets / np.max(offsets)
        offsets = offsets.astype(theano.config.floatX)
        self.grid_offsets = T.constant(offsets)
        # make coordinate vectors for x and y location in the image.
        #   -- image coordinates for the smallest dimension range over
        #      [-init_scale....init_scale], and coordinates for the largest
        #      dimension are at the same scale, but over a larger range.
        x_coords = (np.arange(img_width) - (img_width / 2.0) + 0.5)
        y_coords = (np.arange(img_height) - (img_height / 2.0) + 0.5)
        rescale = min(np.max(x_coords), np.max(y_coords))
        x_coords = (init_scale / rescale) * x_coords
        y_coords = (init_scale / rescale) * y_coords
        x_coords = x_coords.astype(theano.config.floatX)
        y_coords = y_coords.astype(theano.config.floatX)
        self.img_x = T.constant(x_coords)
        self.img_y = T.constant(y_coords)
        return

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
        tol = 1e-3
        # construct x and y coordinates for the grid points
        grid_x = center_x.dimshuffle([0, 'x']) + \
                (delta.dimshuffle([0, 'x']) * self.grid_offsets)
        grid_y = center_y.dimshuffle([0, 'x']) + \
                (delta.dimshuffle([0, 'x']) * self.grid_offsets)

        # construct unnormalized attention weights for each grid point
        FX = T.exp( -(self.img_x - grid_x.dimshuffle([0,1,'x']))**2. / \
                   (2. * sigma.dimshuffle([0,'x','x'])**2.) )
        FY = T.exp( -(self.img_y - grid_y.dimshuffle([0,1,'x']))**2. / \
                   (2. * sigma.dimshuffle([0,'x','x'])**2.) )

        # normalize the attention weights (1 / (sigma * sqrt(2*pi)))
        #Z = sigma.dimshuffle([0,'x','x'])**(-1.0) * (1.0 / 6.283**0.5)
        #FX = Z * FX
        #FY = Z * FY
        FX = FX / (FX.sum(axis=-1).dimshuffle(0, 1, 'x') + tol)
        FY = FY / (FY.sum(axis=-1).dimshuffle(0, 1, 'x') + tol)
        return FY, FX


    def read(self, images, center_y, center_x, delta, sigma):
        """
        Extract a batch of attention windows from the given images.

        Parameters
        ----------
        images : :class:`~tensor.TensorVariable`
            Batch of images with shape (batch_size x img_size). Internally it
            will be reshaped to a (batch_size, img_height, img_width)-shaped
            stack of images.
        center_y : :class:`~tensor.TensorVariable`
            Center coordinates for the attention window.
            Expected shape: (batch_size,)
        center_x : :class:`~tensor.TensorVariable`
            Center coordinates for the attention window.
            Expected shape: (batch_size,)
        delta : :class:`~tensor.TensorVariable`
            Distance between extracted grid points.
            Expected shape: (batch_size,)
        sigma : :class:`~tensor.TensorVariable`
            Std. dev. for Gaussian readout kernel.
            Expected shape: (batch_size,)

        Returns
        -------
        windows : :class:`~tensor.TensorVariable`
            extracted windows of shape: (batch_size x N**2)
        """
        N = self.N
        batch_size = images.shape[0]

        # Reshape input into proper 2d images
        I = images.reshape( (batch_size, self.img_height, self.img_width) )

        # Get separable filterbank
        FY, FX = self.filterbank_matrices(center_y, center_x, delta, sigma)

        # apply to the batch of images
        W = my_batched_dot(my_batched_dot(FY, I), FX.transpose([0,2,1]))
        return W.reshape((batch_size, N*N))

    def write(self, windows, center_y, center_x, delta, sigma):
        """
        Write a batch of windows into full sized images.

        Parameters
        ----------
        windows : :class:`~tensor.TensorVariable`
            Batch of images with shape (batch_size x N*N). Internally it
            will be reshaped to a (batch_size, N, N)-shaped
            stack of images.
        center_y : :class:`~tensor.TensorVariable`
            Center coordinates for the attention window.
            Expected shape: (batch_size,)
        center_x : :class:`~tensor.TensorVariable`
            Center coordinates for the attention window.
            Expected shape: (batch_size,)
        delta : :class:`~tensor.TensorVariable`
            Distance between extracted grid points.
            Expected shape: (batch_size,)
        sigma : :class:`~tensor.TensorVariable`
            Std. dev. for Gaussian readout kernel.
            Expected shape: (batch_size,)

        Returns
        -------
        images : :class:`~tensor.TensorVariable`
            extracted windows of shape: (batch_size x img_height*img_width)
        """
        N = self.N
        batch_size = windows.shape[0]

        # Reshape input into proper 2d windows
        W = windows.reshape( (batch_size, N, N) )

        # Get separable filterbank
        FY, FX = self.filterbank_matrices(center_y, center_x, delta, sigma)

        # apply...
        I = my_batched_dot(my_batched_dot(FY.transpose([0,2,1]), W), FX)

        return I.reshape( (batch_size, self.img_height*self.img_width) )

    def nn2att(self, l):
        """
        Convert neural-net outputs to attention parameters

        Parameters
        ----------
        layer : :class:`~tensor.TensorVariable`
            A batch of neural net outputs with shape (batch_size x 5)

        Returns
        -------
        center_y : :class:`~tensor.TensorVariable`
        center_x : :class:`~tensor.TensorVariable`
        delta : :class:`~tensor.TensorVariable`
        sigma : :class:`~tensor.TensorVariable`
        gamma : :class:`~tensor.TensorVariable`
        """
        # split up the input matrix. this gives back vectors, not columns!
        center_y  = l[:,0]
        center_x  = l[:,1]
        log_delta = l[:,2]
        log_sigma = l[:,3]
        log_gamma = l[:,4]
        # delta, sigma, and gamma are constrained to be non-negative. we'll
        # also "damp" their behavior by rescaling prior to exponential.
        delta = T.exp(log_delta/4.0)
        sigma = T.exp(log_sigma/4.0)
        gamma = 2.0 * T.exp(log_gamma/4.0)
        return center_y, center_x, delta, sigma, gamma


#=============================================================================

class ZoomableAttention1d(object):
    def __init__(self, input_dim, N, init_scale=2.0):
        """
        A zoomable attention window for 1-dimensional inputs.

        Parameters
        ----------
        input_dim : int
            length of the input vectors
        N :
            length of the attention window
        init_scale :
            initial scaling for inputs vs. attention window
        """
        self.input_dim = input_dim
        self.N = N
        self.init_scale = init_scale
        # make offsets for internal dispersement of grid points.
        #   -- internal grid coordinates range over [-1...+1]
        offsets = np.arange(N) - (N / 2.0) + 0.5
        offsets = offsets / np.max(offsets)
        offsets = offsets.astype(theano.config.floatX)
        self.grid_offsets = T.constant(offsets)
        # make coordinate vectors for location in the input.
        #   -- coordinates for the smallest dimension are scaled to range over
        #      [-init_scale....init_scale].
        x_coords = (np.arange(input_dim) - (input_dim / 2.0) + 0.5)
        x_coords = (init_scale / np.max(x_coords)) * x_coords
        x_coords = x_coords.astype(theano.config.floatX)
        self.x_coords = T.constant(x_coords)
        return

    def filterbank_matrix(self, center_x, delta, sigma):
        """
        Create a Fx

        Parameters
        ----------
        center_x : T.vector (shape: batch_size)
                   Y and X center coordinates for the attention window
        delta : T.vector (shape: batch_size)
        sigma : T.vector (shape: batch_size)

        Returns
        -------
            FX : N gaussian blob filters for each input in batch
                 -- shape: (batch_size, N, input_dim)
        """
        tol = 1e-3
        # construct x and y coordinates for the grid points
        #   -- grid_x.shape = (batch_size, N)
        grid_x = center_x.dimshuffle([0, 'x']) + \
                (delta.dimshuffle([0, 'x']) * self.grid_offsets)
        # construct unnormalized attention weights for each grid point
        FX = T.exp( -(self.x_coords - grid_x.dimshuffle([0,1,'x']))**2. / \
                   (2. * sigma.dimshuffle([0,'x','x'])**2.) )
        # normalize the attention weights (1 / (sigma * sqrt(2*pi)))
        #Z = sigma.dimshuffle([0,'x','x'])**(-1.0) * (1.0 / 6.283**0.5)
        #FX = Z * FX
        FX = FX / (FX.sum(axis=-1).dimshuffle(0, 1, 'x') + tol)
        return FX


    def read(self, inputs, center_x, delta, sigma):
        """
        Extract a batch of attention windows from the given inputs.

        Parameters
        ----------
        inputs : :class:`~tensor.TensorVariable`
            Batch of inputs to read from.
            Expected shape: (batch_size, input_dim)
        center_x : :class:`~tensor.TensorVariable`
            Center coordinates for the attention window.
            Expected shape: (batch_size,)
        delta : :class:`~tensor.TensorVariable`
            Distance between extracted grid points.
            Expected shape: (batch_size,)
        sigma : :class:`~tensor.TensorVariable`
            Std. dev. for Gaussian readout kernel.
            Expected shape: (batch_size,)

        Returns
        -------
        windows : :class:`~tensor.TensorVariable`
                  extracted windows of shape: (batch_size x N)
        """
        N = self.N
        batch_size = inputs.shape[0]
        # Get separable filterbank
        FX = self.filterbank_matrix(center_x, delta, sigma)
        # apply to the batch of inputs
        _W = FX * inputs.dimshuffle(0,'x',1)
        W = _W.sum(axis=-2)
        return W

    def write(self, windows, center_x, delta, sigma):
        """
        Write a batch of windows into full sized inputs.

        Parameters
        ----------
        windows : :class:`~tensor.TensorVariable`
                  Batch of values to write to empty inputs.
                  Expected shape: (batch_size, N)
        center_x : :class:`~tensor.TensorVariable`
                   Center coordinates for the attention window.
                   Expected shape: (batch_size,)
        delta : :class:`~tensor.TensorVariable`
                Distance between extracted grid points.
                Expected shape: (batch_size,)
        sigma : :class:`~tensor.TensorVariable`
                Std. dev. for Gaussian readout kernel.
                Expected shape: (batch_size,)

        Returns
        -------
        outputs : :class:`~tensor.TensorVariable`
                  extracted windows of shape: (batch_size x input_dim)
        """
        # Get separable filterbank
        FX = self.filterbank_matrix(center_x, delta, sigma)
        # compute the output -- we can figure this out with "shape analysis"
        #   windows.shape = (batch_size, N)
        #   FX.shape = (batch_size, N, input_dim)
        #   outputs.shape = (batch_size, input_dim)
        # so... do dimshuffle on windows, then do elementwise multiply with FX,
        # and then aggregate over the length-N axis by a sum...
        _outputs = FX * windows.dimshuffle(0, 1, 'x')
        outputs = _outputs.sum(axis=-2)
        return outputs

    def nn2att(self, l):
        """
        Convert neural-net outputs to attention parameters

        Parameters
        ----------
        layer : :class:`~tensor.TensorVariable`
            A batch of neural net outputs with shape (batch_size x 4)

        Returns
        -------
        center_x : :class:`~tensor.TensorVariable`
        delta : :class:`~tensor.TensorVariable`
        sigma : :class:`~tensor.TensorVariable`
        gamma : :class:`~tensor.TensorVariable`
        """
        # split up the input matrix. this gives back vectors, not columns!
        center_x  = l[:,0]
        log_delta = l[:,1]
        log_sigma = l[:,2]
        log_gamma = l[:,3]
        # delta, sigma, and gamma are constrained to be non-negative. we'll
        # also "damp" their behavior by rescaling prior to exponential.
        delta = T.exp(log_delta/4.0)
        sigma = T.exp(log_sigma/4.0)
        gamma = 2.0 * T.exp(log_gamma/4.0)
        return center_x, delta, sigma, gamma




###############
###############
### TESTING ###
###############
###############

if __name__ == "__main__":
    input_dim = 10
    N = 3
    zoom_1d = ZoomableAttention1d(input_dim=input_dim, N=N, init_scale=2.0)
    _center_x = T.vector()
    _delta = T.vector()
    _sigma = T.vector()
    _inputs = T.matrix()
    _windows = T.matrix()
    _F = zoom_1d.filterbank_matrix(_center_x, _delta, _sigma)
    _R = zoom_1d.read(_inputs, _center_x, _delta, _sigma)
    _W = zoom_1d.write(_windows, _center_x, _delta, _sigma)
    test_filterbank_matrix = theano.function( \
            inputs=[_center_x, _delta, _sigma], outputs=_F)
    test_read = theano.function( \
            inputs=[_inputs, _center_x, _delta, _sigma], outputs=_R)
    test_write = theano.function( \
            inputs=[_windows, _center_x, _delta, _sigma], outputs=_W)

    batch_size = 5
    inputs = npr.rand(batch_size, input_dim).astype(theano.config.floatX)
    windows = npr.rand(batch_size, N).astype(theano.config.floatX)
    center_x = np.zeros((batch_size,)).astype(theano.config.floatX)
    delta = np.ones((batch_size,)).astype(theano.config.floatX)
    sigma = 0.25 * delta
    print("Testing filterbank matrix construction...")
    F = test_filterbank_matrix(center_x, delta, sigma)
    print("F.shape: {}".format(F.shape))
    print("Testing reading function...")
    R = test_read(inputs, center_x, delta, sigma)
    print("R.shape: {}".format(R.shape))
    print("Testing writing function...")
    W = test_write(windows, center_x, delta, sigma)
    print("W.shape: {}".format(W.shape))
    print("Done.")
