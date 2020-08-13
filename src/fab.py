# Copyright (c) 2020, Shane Colburn, University of Washington
# This file is part of rcwa_tf
# Written by Shane Colburn (Email: scolbur2@uw.edu)

import tensorflow as tf
import numpy as np

    
def convolve_density_with_blur(density, blur):
  '''
    This function computes the convolution of two inputs to return a blurred
    density function.
    Args:
        density: A `tf.Tensor` of dtype `tf.float32` and shape `(1, pixelsX, 
        pixelsY, Nlayers - 1, Nx, Ny)` specifying a density function with values
        in the range from 0 to 1 on the Nx and Ny dimensions.

        blur: A `tf.Tensor` of dtype `tf.float32` and shape `(1, pixelsX, 
        pixelsY, Nlayers - 1, Nx, Ny)` specifying a blur function on the Nx and
        Ny dimensions.
    Returns:
        A `tf.Tensor` of dtype `tf.float32` and shape `(1, pixelsX, pixelsY, 
        Nlayers - 1, Nx, Ny)` specifying the blurred density.
    '''

  _, _, _, _, Nx, Ny = density.shape

  # Padding to accomodate linear convolution.
  paddings = ((0, 0), (0, 0), (0, 0), (0, 0), (Nx // 2, Nx // 2), (Ny // 2, Ny // 2))
  density_padded = tf.pad(density, paddings = paddings)
  density_padded = tf.cast(density_padded, dtype = tf.complex64)

  blur_padded = tf.pad(blur, paddings = paddings)
  blur_padded = tf.cast(blur_padded, dtype = tf.complex64)

  # Perform the convolution in the Fourier domain and return the image.
  convolved = tf.signal.ifftshift(tf.signal.ifft2d(tf.signal.fft2d(density_padded) * tf.signal.fft2d(blur_padded)), axes = (4, 5))
  x_low = Nx // 2
  x_high = x_low + Nx
  y_low = Ny // 2
  y_high = y_low + Ny
  convolved_cropped = convolved[:, :, :, :, x_low : x_high, y_low : y_high]
    
  return tf.abs(convolved_cropped)


def blur_unit_cell(eps_r, params):
  '''
    This function blurs a unit cell to remove high spatial frequency features.
    Args:
        eps_r: A `tf.Tensor` of shape `(1, pixelsX, pixelsY, Nlayer - 1, Nx, Ny)`
        and dtype `tf.float32` specifying the permittivity at each point in the 
        unit cell grid.

        params: A `dict` containing simulation and optimization settings.
    Returns:
        A `tf.Tensor` of dtype `tf.float32` and shape `(1, pixelsX, pixelsY, 
        Nlayers - 1, Nx, Ny)` specifying the blurred real space permittivity
        on a Cartesian grid.
    '''
  # Define the cartesian cross section.
  dx = params['Lx'] / params['Nx'] # grid resolution along x
  dy = params['Ly'] / params['Ny'] # grid resolution along y
  xa = np.linspace(0, params['Nx'] - 1, params['Nx']) * dx # x axis array
  xa = xa - np.mean(xa) # center x axis at zero
  ya = np.linspace(0, params['Ny'] - 1, params['Ny']) * dy # y axis vector
  ya = ya - np.mean(ya) # center y axis at zero
  [y_mesh, x_mesh] = np.meshgrid(ya,xa)

  # Blur function.
  R = params['blur_radius']
  circ = tf.cast(x_mesh ** 2 + y_mesh **2 < R ** 2, dtype = tf.float32)
  decay = tf.cast(R - tf.math.sqrt(x_mesh ** 2 + y_mesh ** 2), dtype = tf.float32)
  weight = circ * decay
  weight = weight[tf.newaxis, tf.newaxis, tf.newaxis, tf.newaxis, :, :]
  weight = weight / tf.math.reduce_sum(weight)

  # Blur the unit cell permittivity.
  density = (eps_r - params['eps_min']) / (params['eps_max'] - params['eps_min'])
  density_blurred = convolve_density_with_blur(density, weight)

  return density_blurred * (params['eps_max'] - params['eps_min']) + params['eps_min']


def threshold(eps_r, params):
  '''
    This function applies a non-differentiable threshold operation to snap a 
    design to binary permittivity values.
    Args:
        eps_r: A `tf.Tensor` of shape `(1, pixelsX, pixelsY, Nlayer - 1, Nx, Ny)`
        and dtype `tf.float32` specifying the permittivity at each point in the 
        unit cell grid.

        params: A `dict` containing simulation and optimization settings.
    Returns:
        A `tf.Tensor` of dtype `tf.float32` and shape `(1, pixelsX, pixelsY, 
        Nlayers - 1, Nx, Ny)` specifying the binarized permittivity.
    '''
  # Apply the threshold.
  eps_thresh = eps_r.numpy()
  eps_thresh = (eps_thresh - params['eps_min']) / (params['eps_max'] - params['eps_min'])
  eps_thresh = eps_thresh > 0.5
  eps_thresh = eps_thresh * (params['eps_max'] - params['eps_min']) + params['eps_min']

  return tf.convert_to_tensor(eps_thresh, dtype = tf.float32)


def binary_push(eps_r, weight, params):
  '''
    This function applies a differentiable threshold operation that pushes
    permittivity values towards a binary structure by means of a sigmoid.
    Args:
        eps_r: A `tf.Tensor` of shape `(1, pixelsX, pixelsY, Nlayer - 1, Nx, Ny)`
        and dtype `tf.float32` specifying the permittivity at each point in the 
        unit cell grid.

        weight: A `float` specifying the coefficient of the sigmoid argument that
        specifices how hard to "push" the permittivity to a binary structure.
        The higher the weight, the more binary the structure will be.

        params: A `dict` containing simulation and optimization settings.
    Returns:
        A `tf.Tensor` of dtype `tf.float32` and shape `(1, pixelsX, pixelsY, 
        Nlayers - 1, Nx, Ny)` specifying the permittivity after pushing the
        permittivity closer to a binary structure.
    '''
  density = (eps_r - params['eps_min']) / (params['eps_max'] - params['eps_min'])
  density_pushed = tf.math.sigmoid(weight * (density - 0.5))
  eps_pushed = density_pushed * (params['eps_max'] - params['eps_min']) + params['eps_min']
  return tf.convert_to_tensor(eps_pushed, dtype = tf.float32)
