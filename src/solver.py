import tensorflow as tf
import numpy as np
import rcwa_utils
import tensor_utils


def initialize_params(wavelengths = [632.0],
                      thetas = [0.0],
                      phis = [0.0],
                      pte = [1.0],
                      ptm = [0.0],
                      pixelsX = 1,
                      pixelsY = 1,
                      erd = 6.76,
                      ers = 2.25,
                      PQ = [11, 11],
                      Lx = 0.7 * 632.0,
                      Ly = 0.7 * 632.0,
                      L = [632.0, 632.0],
                      Nx = 512,
                      eps_min = 1.0,
                      eps_max = 12.11,
                      blur_radius = 100.0):
  '''
    Initializes simulation parameters and hyperparameters.
    Args:
        wavelengths: A `list` of dtype `float` and length `batchSize` specifying
        the set of wavelengths over which to optimize.

        thetas: A `list` of dtype `float` and length `batchSize` specifying
        the set of polar angles over which to optimize.

        phis: A `list` of dtype `float` and length `batchSize` specifying the 
        set of azimuthal angles over which to optimize.

        pte: A `list` of dtype `float` and length `batchSize` specifying the set
        of TE polarization component magnitudes over which to optimize. A 
        magnitude of 0.0 means no TE component. Under normal incidence, the TE 
        polarization is parallel to the y-axis.

        ptm: A `list` of dtype `float` and length `batchSize` specifying the set
        of TM polarization component magnitudes over which to optimize. A 
        magnitude of 0.0 means no TM component. Under normal incidence, the TM 
        polarization is parallel to the x-axis.

        pixelsX: An `int` specifying the x dimension of the metasurface in
        pixels that are of width `params['Lx']`.

        pixelsY: An `int` specifying the y dimension of the metasurface in
        pixels that are of width `params['Ly']`.

        erd: A `float` specifying the relative permittivity of the non-vacuum,
        constituent material of the device layer for shape optimizations.

        ers: A `float` specifying the relative permittivity of the substrate
        layer.

        PQ: A `list` of dtype `int` and length 2 specifying the number of 
        Fourier harmonics in the x and y directions. The numbers should be odd
        values.

        Lx: A `float` specifying the unit cell pitch in the x direction in
        nanometers.

        Ly: A `float` specifying the unit cell pitch in the y direction in
        nanometers.

        L: A `list` of dtype `float` specifying the layer thicknesses in 
        nanometers.

        Nx: An `int` specifying the number of sample points along the x 
        direction in the unit cell.

        eps_min: A `float` specifying the minimum allowed permittivity for 
        topology optimizations.

        eps_max: A `float` specifying the maximum allowed permittivity for 
        topology optimizations.

        blur_radius: A `float` specifying the radius of the blur function with 
        which a topology optimized permittivity density should be convolved.
    Returns:
        params: A `dict` containing simulation and optimization settings.
  '''

  # Define the `params` dictionary.
  params = dict({})

  # Units and tensor dimensions.
  params['nanometers'] = 1E-9
  params['degrees'] = np.pi / 180
  params['batchSize'] = len(wavelengths)
  params['pixelsX'] = pixelsX
  params['pixelsY'] = pixelsY
  params['Nlay'] = len(L)

  # Simulation tensor shapes.
  batchSize = params['batchSize']
  simulation_shape = (batchSize, pixelsX, pixelsY)

  # Batch parameters (wavelength, incidence angle, and polarization).
  lam0 = params['nanometers'] * tf.convert_to_tensor(wavelengths, dtype = tf.float32)
  lam0 = lam0[:, tf.newaxis, tf.newaxis, tf.newaxis, tf.newaxis, tf.newaxis]
  lam0 = tf.tile(lam0, multiples = (1, pixelsX, pixelsY, 1, 1, 1))
  params['lam0'] = lam0

  theta = params['degrees'] * tf.convert_to_tensor(thetas, dtype = tf.float32)
  theta = theta[:, tf.newaxis, tf.newaxis, tf.newaxis, tf.newaxis, tf.newaxis]
  theta = tf.tile(theta, multiples = (1, pixelsX, pixelsY, 1, 1, 1))
  params['theta'] = theta

  phi = params['degrees'] * tf.convert_to_tensor(phis, dtype = tf.float32)
  phi = phi[:, tf.newaxis, tf.newaxis, tf.newaxis, tf.newaxis, tf.newaxis]
  phi = tf.tile(phi, multiples = (1, pixelsX, pixelsY, 1, 1, 1))
  params['phi'] = phi

  pte = tf.convert_to_tensor(pte, dtype = tf.complex64)
  pte = pte[:, tf.newaxis, tf.newaxis, tf.newaxis]
  pte = tf.tile(pte, multiples = (1, pixelsX, pixelsY, 1))
  params['pte'] = pte

  ptm = tf.convert_to_tensor(ptm, dtype = tf.complex64)
  ptm = ptm[:, tf.newaxis, tf.newaxis, tf.newaxis]
  ptm = tf.tile(ptm, multiples = (1, pixelsX, pixelsY, 1))
  params['ptm'] = ptm

  # Device parameters.
  params['ur1'] = 1.0 # permeability in reflection region
  params['er1'] = 1.0 # permittivity in reflection region
  params['ur2'] = 1.0 # permeability in transmission region
  params['er2'] = 1.0 # permittivity in transmission region
  params['urd'] = 1.0 # permeability of device
  params['erd'] = erd # permittivity of device
  params['urs'] = 1.0 # permeability of substrate
  params['ers'] = ers # permittivity of substrate
  params['Lx'] = Lx * params['nanometers'] # period along x
  params['Ly'] = Ly * params['nanometers'] # period along y
  length_shape = (1, 1, 1, params['Nlay'], 1, 1)
  L = tf.convert_to_tensor(L, dtype = tf.complex64)
  L = L[tf.newaxis, tf.newaxis, tf.newaxis, :, tf.newaxis, tf.newaxis]
  params['L'] = L * params['nanometers'] #* tf.ones(shape = length_shape, dtype = tf.complex64)
  params['length_min'] = 0.1
  params['length_max'] = 2.0

  # RCWA parameters.
  params['PQ'] = PQ # number of spatial harmonics along x and y
  params['Nx'] = Nx # number of point along x in real-space grid
  if params['PQ'][1] == 1:
    params['Ny'] = 1
  else:
    params['Ny'] = int(np.round(params['Nx'] * params['Ly'] / params['Lx'])) # number of point along y in real-space grid

  # Coefficient for the argument of tf.math.sigmoid() when generating
  # permittivity distributions with geometric parameters.
  params['sigmoid_coeff'] = 1000.0

  # Polynomial order for rectangular resonators definition.
  params['rectangle_power'] = 200

  # Allowed permittivity range.
  params['eps_min'] = eps_min
  params['eps_max'] = eps_max

  # Upsampling for Fourier optics propagation.
  params['upsample'] = 1

  # Duty Cycle limits for gratings.
  params['duty_min'] = 0.1
  params['duty_max'] = 0.9

  # Permittivity density blur radius.
  params['blur_radius'] = blur_radius * params['nanometers']

  return params


def generate_coupled_cylindrical_resonators(r_x, r_y, params):
  '''
    Generates permittivity/permeability for a unit cell comprising 4 coupled
    elliptical resonators.
    Args:
        r_x: A `tf.Tensor` of shape `(1, pixelsX, pixelsY, 4)` specifying the 
        x-axis diameters of the four cylinders.

        r_y: A `tf.Tensor` of shape `(1, pixelsX, pixelsY, 4)` specifying the 
        y-axis diameters of the four cylinders.

        params: A `dict` containing simulation and optimization settings.
    Returns:
        ER_t: A `tf.Tensor` of shape `(batchSize, pixelsX, pixelsY, Nlayer, Nx, Ny)`
        specifying the relative permittivity distribution of the unit cell.

        UR_t: A `tf.Tensor` of shape `(batchSize, pixelsX, pixelsY, Nlayer, Nx, Ny)`
        specifying the relative permeability distribution of the unit cell.
  '''

  # Retrieve simulation size parameters.
  batchSize = params['batchSize']
  pixelsX = params['pixelsX']
  pixelsY = params['pixelsY']
  Nlay = params['Nlay']
  Nx = params['Nx']
  Ny = params['Ny']
  Lx = params['Lx']
  Ly = params['Ly']

  # Initialize relative permeability.
  materials_shape = (batchSize, pixelsX, pixelsY, Nlay, Nx, Ny)
  UR = params['urd'] * np.ones(materials_shape)

  # Define the cartesian cross section.
  dx = Lx / Nx # grid resolution along x
  dy = Ly / Ny # grid resolution along y
  xa = np.linspace(0, Nx - 1, Nx) * dx # x axis array
  xa = xa - np.mean(xa) # center x axis at zero
  ya = np.linspace(0, Ny - 1, Ny) * dy # y axis vector
  ya = ya - np.mean(ya) # center y axis at zero
  [y_mesh, x_mesh] = np.meshgrid(ya,xa)

  # Convert to tensors and expand and tile to match the simulation shape.
  y_mesh = tf.convert_to_tensor(y_mesh, dtype = tf.float32)
  y_mesh = y_mesh[tf.newaxis, tf.newaxis, tf.newaxis, tf.newaxis, :, :]
  y_mesh = tf.tile(y_mesh, multiples = (batchSize, pixelsX, pixelsY, 1, 1, 1))
  x_mesh = tf.convert_to_tensor(x_mesh, dtype = tf.float32)
  x_mesh = x_mesh[tf.newaxis, tf.newaxis, tf.newaxis, tf.newaxis, :, :]
  x_mesh = tf.tile(x_mesh, multiples = (batchSize, pixelsX, pixelsY, 1, 1, 1))

  # Nanopost centers.
  c1_x = -Lx / 4
  c1_y = -Ly / 4
  c2_x = -Lx / 4
  c2_y = Ly / 4
  c3_x = Lx / 4
  c3_y = -Ly / 4
  c4_x = Lx / 4
  c4_y = Ly / 4

  # Clip the optimization ranges.
  r_x = params['Lx'] * tf.clip_by_value(r_x, clip_value_min = 0.05, clip_value_max = 0.23)
  r_y = params['Ly'] * tf.clip_by_value(r_y, clip_value_min = 0.05, clip_value_max = 0.23)
  r_x = tf.tile(r_x, multiples = (batchSize, 1, 1, 1))
  r_y = tf.tile(r_y, multiples = (batchSize, 1, 1, 1))
  r_x = r_x[:, :, :, tf.newaxis, tf.newaxis, tf.newaxis, :]
  r_y = r_y[:, :, :, tf.newaxis, tf.newaxis, tf.newaxis, :]

  # Calculate the nanopost boundaries.
  c1 = 1 - ((x_mesh - c1_x) / r_x[:, :, :, :, :, :, 0]) ** 2 - ((y_mesh - c1_y) / r_y[:, :, :, :, :, :, 0]) ** 2
  c2 = 1 - ((x_mesh - c2_x) / r_x[:, :, :, :, :, :, 1]) ** 2 - ((y_mesh - c2_y) / r_y[:, :, :, :, :, :, 1]) ** 2
  c3 = 1 - ((x_mesh - c3_x) / r_x[:, :, :, :, :, :, 2]) ** 2 - ((y_mesh - c3_y) / r_y[:, :, :, :, :, :, 2]) ** 2
  c4 = 1 - ((x_mesh - c4_x) / r_x[:, :, :, :, :, :, 3]) ** 2 - ((y_mesh - c4_y) / r_y[:, :, :, :, :, :, 3]) ** 2

  # Build device layer.
  ER_c1 = tf.math.sigmoid(params['sigmoid_coeff'] * c1)
  ER_c2 = tf.math.sigmoid(params['sigmoid_coeff'] * c2)
  ER_c3 = tf.math.sigmoid(params['sigmoid_coeff'] * c3)
  ER_c4 = tf.math.sigmoid(params['sigmoid_coeff'] * c4)
  ER_t = 1 + (params['erd'] - 1) * (ER_c1 + ER_c2 + ER_c3 + ER_c4)

  # Build substrate and concatenate along the layers dimension.
  device_shape = (batchSize, pixelsX, pixelsY, 1, Nx, Ny)
  ER_substrate = params['ers'] * tf.ones(device_shape, dtype = tf.float32)
  ER_t = tf.concat(values = [ER_t, ER_substrate], axis = 3)

  # Cast to complex for subsequent calculations.
  ER_t = tf.cast(ER_t, dtype = tf.complex64)
  UR_t = tf.convert_to_tensor(UR, dtype = tf.float32)
  UR_t = tf.cast(UR_t, dtype = tf.complex64)

  return ER_t, UR_t


def generate_coupled_rectangular_resonators(r_x, r_y, params):
  '''
    Generates permittivity/permeability for a unit cell comprising 4 coupled
    rectangular cross section scatterers.
    Args:
        r_x: A `tf.Tensor` of shape `(1, pixelsX, pixelsY, 4)` specifying the 
        x-axis widths of the four rectangles.

        r_y: A `tf.Tensor` of shape `(1, pixelsX, pixelsY, 4)` specifying the 
        y-axis widths of the four rectangles.

        params: A `dict` containing simulation and optimization settings.
    Returns:
        ER_t: A `tf.Tensor` of shape `(batchSize, pixelsX, pixelsY, Nlayer, Nx, Ny)`
        specifying the relative permittivity distribution of the unit cell.

        UR_t: A `tf.Tensor` of shape `(batchSize, pixelsX, pixelsY, Nlayer, Nx, Ny)`
        specifying the relative permeability distribution of the unit cell.
  '''

  # Retrieve simulation size parameters.
  batchSize = params['batchSize']
  pixelsX = params['pixelsX']
  pixelsY = params['pixelsY']
  Nlay = params['Nlay']
  Nx = params['Nx']
  Ny = params['Ny']
  Lx = params['Lx']
  Ly = params['Ly']

  # Initialize relative permeability.
  materials_shape = (batchSize, pixelsX, pixelsY, Nlay, Nx, Ny)
  UR = params['urd'] * np.ones(materials_shape)

  # Define the cartesian cross section.
  dx = Lx / Nx # grid resolution along x
  dy = Ly / Ny # grid resolution along y
  xa = np.linspace(0, Nx - 1, Nx) * dx # x axis array
  xa = xa - np.mean(xa) # center x axis at zero
  ya = np.linspace(0, Ny - 1, Ny) * dy # y axis vector
  ya = ya - np.mean(ya) # center y axis at zero
  [y_mesh, x_mesh] = np.meshgrid(ya,xa)

  # Convert to tensors and expand and tile to match the simulation shape.
  y_mesh = tf.convert_to_tensor(y_mesh, dtype = tf.float32)
  y_mesh = y_mesh[tf.newaxis, tf.newaxis, tf.newaxis, tf.newaxis, :, :]
  y_mesh = tf.tile(y_mesh, multiples = (batchSize, pixelsX, pixelsY, 1, 1, 1))
  x_mesh = tf.convert_to_tensor(x_mesh, dtype = tf.float32)
  x_mesh = x_mesh[tf.newaxis, tf.newaxis, tf.newaxis, tf.newaxis, :, :]
  x_mesh = tf.tile(x_mesh, multiples = (batchSize, pixelsX, pixelsY, 1, 1, 1))

  # Nanopost centers.
  c1_x = -Lx / 4
  c1_y = -Ly / 4
  c2_x = -Lx / 4
  c2_y = Ly / 4
  c3_x = Lx / 4
  c3_y = -Ly / 4
  c4_x = Lx / 4
  c4_y = Ly / 4

  # Nanopost width ranges.
  r_x = params['Lx'] * tf.clip_by_value(r_x, clip_value_min = 0.05, clip_value_max = 0.23)
  r_y = params['Ly'] * tf.clip_by_value(r_y, clip_value_min = 0.05, clip_value_max = 0.23)
  r_x = tf.tile(r_x, multiples = (batchSize, 1, 1, 1))
  r_y = tf.tile(r_y, multiples = (batchSize, 1, 1, 1))
  r_x = r_x[:, :, :, tf.newaxis, tf.newaxis, tf.newaxis, :]
  r_y = r_y[:, :, :, tf.newaxis, tf.newaxis, tf.newaxis, :]

  # Calculate the nanopost boundaries.
  c1 = 1 - ((x_mesh - c1_x) / r_x[:, :, :, :, :, :, 0]) ** params['rectangle_power'] - ((y_mesh - c1_y) / r_y[:, :, :, :, :, :, 0]) ** params['rectangle_power']
  c2 = 1 - ((x_mesh - c2_x) / r_x[:, :, :, :, :, :, 1]) ** params['rectangle_power'] - ((y_mesh - c2_y) / r_y[:, :, :, :, :, :, 1]) ** params['rectangle_power']
  c3 = 1 - ((x_mesh - c3_x) / r_x[:, :, :, :, :, :, 2]) ** params['rectangle_power'] - ((y_mesh - c3_y) / r_y[:, :, :, :, :, :, 2]) ** params['rectangle_power']
  c4 = 1 - ((x_mesh - c4_x) / r_x[:, :, :, :, :, :, 3]) ** params['rectangle_power'] - ((y_mesh - c4_y) / r_y[:, :, :, :, :, :, 3]) ** params['rectangle_power']

  # Build device layer.
  ER_c1 = tf.math.sigmoid(params['sigmoid_coeff'] * c1)
  ER_c2 = tf.math.sigmoid(params['sigmoid_coeff'] * c2)
  ER_c3 = tf.math.sigmoid(params['sigmoid_coeff'] * c3)
  ER_c4 = tf.math.sigmoid(params['sigmoid_coeff'] * c4)
  ER_t = 1 + (params['erd'] - 1) * (ER_c1 + ER_c2 + ER_c3 + ER_c4)

  # Build substrate and concatenate along the layers dimension.
  device_shape = (batchSize, pixelsX, pixelsY, 1, Nx, Ny)
  ER_substrate = params['ers'] * tf.ones(device_shape, dtype = tf.float32)
  ER_t = tf.concat(values = [ER_t, ER_substrate], axis = 3)

  # Cast to complex for subsequent calculations.
  ER_t = tf.cast(ER_t, dtype = tf.complex64)
  UR_t = tf.convert_to_tensor(UR, dtype = tf.float32)
  UR_t = tf.cast(UR_t, dtype = tf.complex64)

  return ER_t, UR_t


def generate_rectangular_resonators(r_x, r_y, params):
  '''
    Generates permittivity/permeability for a unit cell comprising a single, 
    centered rectangular cross section scatterer.

    Args:
        r_x: A `tf.Tensor` of shape `(1, pixelsX, pixelsY, 1)` specifying the 
        x-axis widths of the rectangle.

        r_y: A `tf.Tensor` of shape `(1, pixelsX, pixelsY, 1)` specifying the 
        y-axis widths of the rectangle.

        params: A `dict` containing simulation and optimization settings.
    Returns:
        ER_t: A `tf.Tensor` of shape `(batchSize, pixelsX, pixelsY, Nlayer, Nx, Ny)`
        specifying the relative permittivity distribution of the unit cell.

        UR_t: A `tf.Tensor` of shape `(batchSize, pixelsX, pixelsY, Nlayer, Nx, Ny)`
        specifying the relative permeability distribution of the unit cell.
  '''

  # Retrieve simulation size parameters.
  batchSize = params['batchSize']
  pixelsX = params['pixelsX']
  pixelsY = params['pixelsY']
  Nlay = params['Nlay']
  Nx = params['Nx']
  Ny = params['Ny']
  Lx = params['Lx']
  Ly = params['Ly']

  # Initialize relative permeability.
  materials_shape = (batchSize, pixelsX, pixelsY, Nlay, Nx, Ny)
  UR = params['urd'] * np.ones(materials_shape)

  # Define the cartesian cross section.
  dx = Lx / Nx # grid resolution along x
  dy = Ly / Ny # grid resolution along y
  xa = np.linspace(0, Nx - 1, Nx) * dx # x axis array
  xa = xa - np.mean(xa) # center x axis at zero
  ya = np.linspace(0, Ny - 1, Ny) * dy # y axis vector
  ya = ya - np.mean(ya) # center y axis at zero
  [y_mesh, x_mesh] = np.meshgrid(ya,xa)

  # Convert to tensors and expand and tile to match the simulation shape.
  y_mesh = tf.convert_to_tensor(y_mesh, dtype = tf.float32)
  y_mesh = y_mesh[tf.newaxis, tf.newaxis, tf.newaxis, tf.newaxis, :, :]
  y_mesh = tf.tile(y_mesh, multiples = (batchSize, pixelsX, pixelsY, 1, 1, 1))
  x_mesh = tf.convert_to_tensor(x_mesh, dtype = tf.float32)
  x_mesh = x_mesh[tf.newaxis, tf.newaxis, tf.newaxis, tf.newaxis, :, :]
  x_mesh = tf.tile(x_mesh, multiples = (batchSize, pixelsX, pixelsY, 1, 1, 1))

  # Limit the optimization ranges.
  r_x = params['Lx'] * tf.clip_by_value(r_x, clip_value_min = 0.05, clip_value_max = 0.46)
  r_y = params['Ly'] * tf.clip_by_value(r_y, clip_value_min = 0.05, clip_value_max = 0.46)
  r_x = tf.tile(r_x, multiples = (batchSize, 1, 1, 1))
  r_y = tf.tile(r_y, multiples = (batchSize, 1, 1, 1))
  r_x = r_x[:, :, :, tf.newaxis, tf.newaxis, tf.newaxis, :]
  r_y = r_y[:, :, :, tf.newaxis, tf.newaxis, tf.newaxis, :]

  r1 = 1 - tf.abs((x_mesh / 2 / r_x[:, :, :, :, :, :, 0]) - (y_mesh / 2 / r_y[:, :, :, :, :, :, 0])) - tf.abs((x_mesh / 2 / r_x[:, :, :, :, :, :, 0]) + (y_mesh / 2 / r_y[:, :, :, :, :, :, 0]))

  # Build device layer.
  ER_r1 = tf.math.sigmoid(params['sigmoid_coeff'] * r1)
  ER_t = 1 + (params['erd'] - 1) * ER_r1

  # Build substrate and concatenate along the layers dimension.
  device_shape = (batchSize, pixelsX, pixelsY, 1, Nx, Ny)
  ER_substrate = params['ers'] * tf.ones(device_shape, dtype = tf.float32)
  ER_t = tf.concat(values = [ER_t, ER_substrate], axis = 3)

  # Cast to complex for subsequent calculations.
  ER_t = tf.cast(ER_t, dtype = tf.complex64)
  UR_t = tf.convert_to_tensor(UR, dtype = tf.float32)
  UR_t = tf.cast(UR_t, dtype = tf.complex64)

  return ER_t, UR_t


def generate_elliptical_resonators(r_x, r_y, params):
  '''
    Generates permittivity/permeability for a unit cell comprising a single, 
    centered elliptical cross section scatterer.

    Args:
        r_x: A `tf.Tensor` of shape `(1, pixelsX, pixelsY, 1)` specifying the 
        x-axis diameter of the ellipse.

        r_y: A `tf.Tensor` of shape `(1, pixelsX, pixelsY, 1)` specifying the 
        y-axis diameter of the ellipse.

        params: A `dict` containing simulation and optimization settings.
    Returns:
        ER_t: A `tf.Tensor` of shape `(batchSize, pixelsX, pixelsY, Nlayer, Nx, Ny)`
        specifying the relative permittivity distribution of the unit cell.

        UR_t: A `tf.Tensor` of shape `(batchSize, pixelsX, pixelsY, Nlayer, Nx, Ny)`
        specifying the relative permeability distribution of the unit cell.
  '''

  # Retrieve simulation size parameters.
  batchSize = params['batchSize']
  pixelsX = params['pixelsX']
  pixelsY = params['pixelsY']
  Nlay = params['Nlay']
  Nx = params['Nx']
  Ny = params['Ny']
  Lx = params['Lx']
  Ly = params['Ly']

  # Initialize relative permeability.
  materials_shape = (batchSize, pixelsX, pixelsY, Nlay, Nx, Ny)
  UR = params['urd'] * np.ones(materials_shape)

  # Define the cartesian cross section.
  dx = Lx / Nx # grid resolution along x
  dy = Ly / Ny # grid resolution along y
  xa = np.linspace(0, Nx - 1, Nx) * dx # x axis array
  xa = xa - np.mean(xa) # center x axis at zero
  ya = np.linspace(0, Ny - 1, Ny) * dy # y axis vector
  ya = ya - np.mean(ya) # center y axis at zero
  [y_mesh, x_mesh] = np.meshgrid(ya,xa)

  # Convert to tensors and expand and tile to match the simulation shape.
  y_mesh = tf.convert_to_tensor(y_mesh, dtype = tf.float32)
  y_mesh = y_mesh[tf.newaxis, tf.newaxis, tf.newaxis, tf.newaxis, :, :]
  y_mesh = tf.tile(y_mesh, multiples = (batchSize, pixelsX, pixelsY, 1, 1, 1))
  x_mesh = tf.convert_to_tensor(x_mesh, dtype = tf.float32)
  x_mesh = x_mesh[tf.newaxis, tf.newaxis, tf.newaxis, tf.newaxis, :, :]
  x_mesh = tf.tile(x_mesh, multiples = (batchSize, pixelsX, pixelsY, 1, 1, 1))

  # Limit the optimization ranges.
  r_x = params['Lx'] * tf.clip_by_value(r_x, clip_value_min = 0.05, clip_value_max = 0.46)
  r_y = params['Ly'] * tf.clip_by_value(r_y, clip_value_min = 0.05, clip_value_max = 0.46)
  r_x = tf.tile(r_x, multiples = (batchSize, 1, 1, 1))
  r_y = tf.tile(r_y, multiples = (batchSize, 1, 1, 1))
  r_x = r_x[:, :, :, tf.newaxis, tf.newaxis, tf.newaxis, :]
  r_y = r_y[:, :, :, tf.newaxis, tf.newaxis, tf.newaxis, :]

  # Calculate the ellipse boundary.
  c1 = 1 - (x_mesh / r_x[:, :, :, :, :, :, 0]) ** 2 - (y_mesh / r_y[:, :, :, :, :, :, 0]) ** 2
  
  # Build device layer.
  ER_c1 = tf.math.sigmoid(params['sigmoid_coeff'] * c1)
  ER_t = 1 + (params['erd'] - 1) * ER_c1

  # Build substrate and concatenate along the layers dimension.
  device_shape = (batchSize, pixelsX, pixelsY, 1, Nx, Ny)
  ER_substrate = params['ers'] * tf.ones(device_shape, dtype = tf.float32)
  ER_t = tf.concat(values = [ER_t, ER_substrate], axis = 3)

  # Cast to complex for subsequent calculations.
  ER_t = tf.cast(ER_t, dtype = tf.complex64)
  UR_t = tf.convert_to_tensor(UR, dtype = tf.float32)
  UR_t = tf.cast(UR_t, dtype = tf.complex64)

  return ER_t, UR_t


def generate_cylindrical_nanoposts(duty, params):
  '''
    Generates permittivity/permeability for a unit cell comprising a single, 
    centered circular cross section scatterer.

    Args:
        duty: A `tf.Tensor` of shape `(1, pixelsX, pixelsY, 1)` specifying the 
        duty cycle (diameter / period) of the cylindrical nanopost.

        params: A `dict` containing simulation and optimization settings.
    Returns:
        ER_t: A `tf.Tensor` of shape `(batchSize, pixelsX, pixelsY, Nlayer, Nx, Ny)`
        specifying the relative permittivity distribution of the unit cell.

        UR_t: A `tf.Tensor` of shape `(batchSize, pixelsX, pixelsY, Nlayer, Nx, Ny)`
        specifying the relative permeability distribution of the unit cell.
  '''

  # Retrieve simulation size parameters.
  batchSize = params['batchSize']
  pixelsX = params['pixelsX']
  pixelsY = params['pixelsY']
  Nlay = params['Nlay']
  Nx = params['Nx']
  Ny = params['Ny']

  # Initialize relative permeability.
  materials_shape = (batchSize, pixelsX, pixelsY, Nlay, Nx, Ny)
  UR = params['urd'] * np.ones(materials_shape)

  # Define the cartesian cross section.
  dx = params['Lx'] / Nx # grid resolution along x
  dy = params['Ly'] / Ny # grid resolution along y
  xa = np.linspace(0, Nx - 1, Nx) * dx # x axis array
  xa = xa - np.mean(xa) # center x axis at zero
  ya = np.linspace(0, Ny - 1, Ny) * dy # y axis vector
  ya = ya - np.mean(ya) # center y axis at zero
  [y_mesh, x_mesh] = np.meshgrid(ya,xa)

  # Convert to tensors and expand and tile to match the simulation shape.
  y_mesh = tf.convert_to_tensor(y_mesh, dtype = tf.float32)
  y_mesh = y_mesh[tf.newaxis, tf.newaxis, tf.newaxis, tf.newaxis, :, :]
  y_mesh = tf.tile(y_mesh, multiples = (batchSize, pixelsX, pixelsY, 1, 1, 1))
  x_mesh = tf.convert_to_tensor(x_mesh, dtype = tf.float32)
  x_mesh = x_mesh[tf.newaxis, tf.newaxis, tf.newaxis, tf.newaxis, :, :]
  x_mesh = tf.tile(x_mesh, multiples = (batchSize, pixelsX, pixelsY, 1, 1, 1))

  # Build device layer.
  a = tf.clip_by_value(duty, clip_value_min = params['duty_min'], clip_value_max = params['duty_max'])
  a = a[:, :, :, tf.newaxis, tf.newaxis, tf.newaxis]
  a = tf.tile(a, multiples = (1, 1, 1, 1, Nx, Ny))
  radius = 0.5 * params['Ly'] * a
  sigmoid_arg = (1 - (x_mesh / radius) ** 2 - (y_mesh / radius) ** 2)
  ER_t = tf.math.sigmoid(params['sigmoid_coeff'] * sigmoid_arg)
  ER_t = 1 + (params['erd'] - 1) * ER_t

  # Build substrate and concatenate along the layers dimension.
  device_shape = (batchSize, pixelsX, pixelsY, 1, Nx, Ny)
  ER_substrate = params['ers'] * tf.ones(device_shape, dtype = tf.float32)
  ER_t = tf.concat(values = [ER_t, ER_substrate], axis = 3)

  # Cast to complex for subsequent calculations.
  ER_t = tf.cast(ER_t, dtype = tf.complex64)
  UR_t = tf.convert_to_tensor(UR, dtype = tf.float32)
  UR_t = tf.cast(UR_t, dtype = tf.complex64)

  return ER_t, UR_t


def generate_stacked_cylindrical_nanoposts(duty, params):
  '''
    Generates permittivity/permeability for a unit cell comprising a stacked
    cylinders.

    Args:
        duty: A `tf.Tensor` of shape `(1, 1, 1, Nlay - 1, 1, 1)` specifying the 
        thicknesses of the cylinders in each layer, excluding the substrate 
        tihckness.

        params: A `dict` containing simulation and optimization settings.
    Returns:
        ER_t: A `tf.Tensor` of shape `(batchSize, pixelsX, pixelsY, Nlayer, Nx, Ny)`
        specifying the relative permittivity distribution of the unit cell.

        UR_t: A `tf.Tensor` of shape `(batchSize, pixelsX, pixelsY, Nlayer, Nx, Ny)`
        specifying the relative permeability distribution of the unit cell.
  '''

  # Retrieve simulation size parameters.
  batchSize = params['batchSize']
  pixelsX = params['pixelsX']
  pixelsY = params['pixelsY']
  Nlay = params['Nlay']
  Nx = params['Nx']
  Ny = params['Ny']

  # Initialize relative permeability.
  materials_shape = (batchSize, pixelsX, pixelsY, Nlay, Nx, Ny)
  UR = params['urd'] * np.ones(materials_shape)

  # Define the cartesian cross section.
  dx = params['Lx'] / Nx # grid resolution along x
  dy = params['Ly'] / Ny # grid resolution along y
  xa = np.linspace(0, Nx - 1, Nx) * dx # x axis array
  xa = xa - np.mean(xa) # center x axis at zero
  ya = np.linspace(0, Ny - 1, Ny) * dy # y axis vector
  ya = ya - np.mean(ya) # center y axis at zero
  [y_mesh, x_mesh] = np.meshgrid(ya,xa)

  # Convert to tensors and expand and tile to match the simulation shape.
  y_mesh = tf.convert_to_tensor(y_mesh, dtype = tf.float32)
  y_mesh = y_mesh[tf.newaxis, tf.newaxis, tf.newaxis, tf.newaxis, :, :]
  y_mesh = tf.tile(y_mesh, multiples = (batchSize, pixelsX, pixelsY, Nlay - 1, 1, 1))
  x_mesh = tf.convert_to_tensor(x_mesh, dtype = tf.float32)
  x_mesh = x_mesh[tf.newaxis, tf.newaxis, tf.newaxis, tf.newaxis, :, :]
  x_mesh = tf.tile(x_mesh, multiples = (batchSize, pixelsX, pixelsY, Nlay - 1, 1, 1))

  # Build device layer.
  a = tf.clip_by_value(duty, clip_value_min = params['duty_min'], clip_value_max = params['duty_max'])
  a = a[:, :, :, :, tf.newaxis, tf.newaxis]
  a = tf.tile(a, multiples = (1, 1, 1, 1, Nx, Ny))
  radius = 0.5 * params['Ly'] * a
  sigmoid_arg = (1 - (x_mesh / radius) ** 2 - (y_mesh / radius) ** 2)
  ER_t = tf.math.sigmoid(params['sigmoid_coeff'] * sigmoid_arg)
  ER_t = 1 + (params['erd'] - 1) * ER_t

  # Build substrate and concatenate along the layers dimension.
  device_shape = (batchSize, pixelsX, pixelsY, 1, Nx, Ny)
  ER_substrate = params['ers'] * tf.ones(device_shape, dtype = tf.float32)
  ER_t = tf.concat(values = [ER_t, ER_substrate], axis = 3)

  # Cast to complex for subsequent calculations.
  ER_t = tf.cast(ER_t, dtype = tf.complex64)
  UR_t = tf.convert_to_tensor(UR, dtype = tf.float32)
  UR_t = tf.cast(UR_t, dtype = tf.complex64)

  return ER_t, UR_t


def generate_rectangular_lines(duty, params):
  '''
    Generates permittivity/permeability for a unit cell comprising a single
    rectangle that spans the full y length and with width defined along the x
    direction.

    Args:
        duty: A `tf.Tensor` of shape `(1, pixelsX, pixelsY)` specifying the duty
        cycle (i.e., width / pitch) along the x direction for the rectangle.

        params: A `dict` containing simulation and optimization settings.
    Returns:
        ER_t: A `tf.Tensor` of shape `(batchSize, pixelsX, pixelsY, Nlayer, Nx, Ny)`
        specifying the relative permittivity distribution of the unit cell.

        UR_t: A `tf.Tensor` of shape `(batchSize, pixelsX, pixelsY, Nlayer, Nx, Ny)`
        specifying the relative permeability distribution of the unit cell.
  '''

  # Retrieve simulation size parameters.
  batchSize = params['batchSize']
  pixelsX = params['pixelsX']
  pixelsY = params['pixelsY']
  Nlay = params['Nlay']
  Nx = params['Nx']
  Ny = params['Ny']

  # Initialize relative permeability.
  materials_shape = (batchSize, pixelsX, pixelsY, Nlay, Nx, Ny)
  UR = params['urd'] * np.ones(materials_shape)

  # Define the cartesian cross section.
  dx = params['Lx'] / Nx # grid resolution along x
  dy = params['Ly'] / Ny # grid resolution along y
  xa = np.linspace(0, Nx - 1, Nx) * dx # x axis array
  xa = xa - np.mean(xa) # center x axis at zero
  ya = np.linspace(0, Ny - 1, Ny) * dy # y axis vector
  ya = ya - np.mean(ya) # center y axis at zero
  [y_mesh, x_mesh] = np.meshgrid(ya,xa)

  # Convert to tensors and expand and tile to match the simulation shape.
  y_mesh = tf.convert_to_tensor(y_mesh, dtype = tf.float32)
  y_mesh = y_mesh[tf.newaxis, tf.newaxis, tf.newaxis, tf.newaxis, :, :]
  y_mesh = tf.tile(y_mesh, multiples = (batchSize, pixelsX, pixelsY, 1, 1, 1))
  x_mesh = tf.convert_to_tensor(x_mesh, dtype = tf.float32)
  x_mesh = x_mesh[tf.newaxis, tf.newaxis, tf.newaxis, tf.newaxis, :, :]
  x_mesh = tf.tile(x_mesh, multiples = (batchSize, pixelsX, pixelsY, 1, 1, 1))

  # Build device layer.
  a = tf.clip_by_value(duty, clip_value_min = params['duty_min'], clip_value_max = params['duty_max'])
  a = a[:, :, :, tf.newaxis, tf.newaxis, tf.newaxis]
  a = tf.tile(a, multiples = (1, 1, 1, 1, Nx, Ny))
  radius = 0.5 * params['Ly'] * a
  sigmoid_arg = 1 - tf.math.abs(x_mesh / radius)
  ER_t = tf.math.sigmoid(params['sigmoid_coeff'] * sigmoid_arg)
  ER_t = 1 + (params['erd'] - 1) * ER_t

  # Build substrate and concatenate along the layers dimension.
  device_shape = (batchSize, pixelsX, pixelsY, 1, Nx, Ny)
  ER_substrate = params['ers'] * tf.ones(device_shape, dtype = tf.float32)
  ER_t = tf.concat(values = [ER_t, ER_substrate], axis = 3)

  # Cast to complex for subsequent calculations.
  ER_t = tf.cast(ER_t, dtype = tf.complex64)
  UR_t = tf.convert_to_tensor(UR, dtype = tf.float32)
  UR_t = tf.cast(UR_t, dtype = tf.complex64)

  return ER_t, UR_t


def generate_plasmonic_cylindrical_nanoposts(duty, params):
  '''
    Generates permittivity/permeability for a unit cell comprising a single, 
    centered circular cross section plasmonic scatterer with a complex-valued
    permittivity.

    Args:
        duty: A `tf.Tensor` of shape `(1, pixelsX, pixelsY, 1)` specifying the 
        duty cycle (diameter / period) of the cylindrical nanopost.

        params: A `dict` containing simulation and optimization settings.
    Returns:
        ER_t: A `tf.Tensor` of shape `(batchSize, pixelsX, pixelsY, Nlayer, Nx, Ny)`
        specifying the relative permittivity distribution of the unit cell.

        UR_t: A `tf.Tensor` of shape `(batchSize, pixelsX, pixelsY, Nlayer, Nx, Ny)`
        specifying the relative permeability distribution of the unit cell.
  '''

  # Retrieve simulation size parameters.
  batchSize = params['batchSize']
  pixelsX = params['pixelsX']
  pixelsY = params['pixelsY']
  Nlay = params['Nlay']
  Nx = params['Nx']
  Ny = params['Ny']

  # Initialize relative permeability.
  materials_shape = (batchSize, pixelsX, pixelsY, Nlay, Nx, Ny)
  UR = params['urd'] * np.ones(materials_shape)

  # Define the cartesian cross section.
  dx = params['Lx'] / Nx # grid resolution along x
  dy = params['Ly'] / Ny # grid resolution along y
  xa = np.linspace(0, Nx - 1, Nx) * dx # x axis array
  xa = xa - np.mean(xa) # center x axis at zero
  ya = np.linspace(0, Ny - 1, Ny) * dy # y axis vector
  ya = ya - np.mean(ya) # center y axis at zero
  [y_mesh, x_mesh] = np.meshgrid(ya,xa)

  # Convert to tensors and expand and tile to match the simulation shape.
  y_mesh = tf.convert_to_tensor(y_mesh, dtype = tf.float32)
  y_mesh = y_mesh[tf.newaxis, tf.newaxis, tf.newaxis, tf.newaxis, :, :]
  y_mesh = tf.tile(y_mesh, multiples = (batchSize, pixelsX, pixelsY, 1, 1, 1))
  x_mesh = tf.convert_to_tensor(x_mesh, dtype = tf.float32)
  x_mesh = x_mesh[tf.newaxis, tf.newaxis, tf.newaxis, tf.newaxis, :, :]
  x_mesh = tf.tile(x_mesh, multiples = (batchSize, pixelsX, pixelsY, 1, 1, 1))

  # Build device layer.
  a = tf.clip_by_value(duty, clip_value_min = params['duty_min'], clip_value_max = params['duty_max'])
  a = a[:, :, :, tf.newaxis, tf.newaxis, tf.newaxis]
  a = tf.tile(a, multiples = (1, 1, 1, 1, Nx, Ny))
  radius = 0.5 * params['Ly'] * a
  sigmoid_arg = (1 - (x_mesh / radius) ** 2 - (y_mesh / radius) ** 2)
  ER_t = tf.math.sigmoid(params['sigmoid_coeff'] * sigmoid_arg)
  ER_t = tf.cast(ER_t, dtype = tf.complex64)
  ER_t = 1 + (params['erd'] - 1) * ER_t

  # Build substrate and concatenate along the layers dimension.
  device_shape = (batchSize, pixelsX, pixelsY, 1, Nx, Ny)
  ER_substrate = params['ers'] * tf.ones(device_shape, dtype = tf.complex64)
  ER_t = tf.concat(values = [ER_t, ER_substrate], axis = 3)

  # Cast to complex for subsequent calculations.
  UR_t = tf.convert_to_tensor(UR, dtype = tf.float32)
  UR_t = tf.cast(UR_t, dtype = tf.complex64)

  return ER_t, UR_t


def generate_arbitrary_epsilon(eps_r, params):
  '''
    Generates permittivity/permeability for a unit cell comprising a continuously
    varying permittivity for each pixel in the Cartesian grid.

    Args:
        eps_r: A `tf.Tensor` of shape `(1, pixelsX, pixelsY, Nlayer - 1, Nx, Ny)`
        and type `tf.float32` specifying the permittivity at each point in the 
        unit cell grid. The `Nlayer - 1` eps_r.shape[3] length corresponds to 
        there being a fixed substrate that is unchanging between iterations.

        params: A `dict` containing simulation and optimization settings.
    Returns:
        ER_t: A `tf.Tensor` of shape `(batchSize, pixelsX, pixelsY, Nlayer, Nx, Ny)`
        specifying the relative permittivity distribution of the unit cell.

        UR_t: A `tf.Tensor` of shape `(batchSize, pixelsX, pixelsY, Nlayer, Nx, Ny)`
        specifying the relative permeability distribution of the unit cell.
  '''

  # Retrieve simulation size parameters.
  batchSize = params['batchSize']
  pixelsX = params['pixelsX']
  pixelsY = params['pixelsY']
  Nlay = params['Nlay']
  Nx = params['Nx']
  Ny = params['Ny']

  # Initialize relative permeability.
  materials_shape = (batchSize, pixelsX, pixelsY, Nlay, Nx, Ny)
  UR = params['urd'] * np.ones(materials_shape)

  # Set the permittivity.
  ER_t = tf.clip_by_value(eps_r, clip_value_min = params['eps_min'], clip_value_max = params['eps_max'])
  ER_t = tf.tile(ER_t, multiples = (batchSize, 1, 1, 1, 1, 1))

  # Build substrate and concatenate along the layers dimension.
  device_shape = (batchSize, pixelsX, pixelsY, 1, Nx, Ny)
  ER_substrate = params['ers'] * tf.ones(device_shape, dtype = tf.float32)
  ER_t = tf.concat(values = [ER_t, ER_substrate], axis = 3)

  # Cast to complex for subsequent calculations.
  ER_t = tf.cast(ER_t, dtype = tf.complex64)
  UR_t = tf.convert_to_tensor(UR, dtype = tf.float32)
  UR_t = tf.cast(UR_t, dtype = tf.complex64)

  return ER_t, UR_t


def make_propagator(params, f):
  '''
    Pre-computes the band-limited angular spectrum propagator for modelling
    free-space propagation for the distance and sampling as specified in `params`.

    Args:
        params: A `dict` containing simulation and optimization settings.

        f: A `float` specifying the focal length, or distance to propagate, in
        meters.
    Returns:
        propagator: a `tf.Tensor` of shape `(batchSize, params['upsample'] * pixelsX,
        params['upsample'] * pixelsY)` and dtype `tf.complex64` defining the 
        reciprocal space, band-limited angular spectrum propagator.
  '''

  # Simulation tensor shape.
  batchSize = params['batchSize']
  pixelsX = params['pixelsX']
  pixelsY = params['pixelsY']
  upsample = params['upsample']

  # Propagator definition.
  k = 2 * np.pi / params['lam0'][:, 0, 0, 0, 0, 0]
  k = k[:, np.newaxis, np.newaxis]
  samp = params['upsample'] * pixelsX
  k = tf.tile(k, multiples = (1, 2 * samp - 1, 2 * samp - 1))
  k = tf.cast(k, dtype = tf.complex64)  
  k_xlist_pos = 2 * np.pi * np.linspace(0, 1 / (2 *  params['Lx'] / params['upsample']), samp)  
  front = k_xlist_pos[-(samp - 1):]
  front = -front[::-1]
  k_xlist = np.hstack((front, k_xlist_pos))
  k_x = np.kron(k_xlist, np.ones((2 * samp - 1, 1)))
  k_x = k_x[np.newaxis, :, :]
  k_y = np.transpose(k_x, axes = [0, 2, 1])
  k_x = tf.convert_to_tensor(k_x, dtype = tf.complex64)
  k_x = tf.tile(k_x, multiples = (batchSize, 1, 1))
  k_y = tf.convert_to_tensor(k_y, dtype = tf.complex64)
  k_y = tf.tile(k_y, multiples = (batchSize, 1, 1))
  k_z_arg = tf.square(k) - (tf.square(k_x) + tf.square(k_y))
  k_z = tf.sqrt(k_z_arg)
  propagator_arg = 1j * k_z * f
  propagator = tf.exp(propagator_arg)

  # Limit transfer function bandwidth to prevent aliasing.
  kx_limit = 2 * np.pi * (((1 / (pixelsX * params['Lx'])) * f) ** 2 + 1) ** (-0.5) / params['lam0'][:, 0, 0, 0, 0, 0]
  kx_limit = tf.cast(kx_limit, dtype = tf.complex64)
  ky_limit = kx_limit
  kx_limit = kx_limit[:, tf.newaxis, tf.newaxis]
  ky_limit = ky_limit[:, tf.newaxis, tf.newaxis]

  # Apply the antialiasing filter.
  ellipse_kx = (tf.square(k_x / kx_limit) + tf.square(k_y / k)).numpy() <= 1
  ellipse_ky = (tf.square(k_x / k) + tf.square(k_y / ky_limit)).numpy() <= 1
  propagator = propagator * ellipse_kx * ellipse_ky

  return propagator


def propagate(field, propagator, upsample):
  '''
    Propagates a batch of input fields to a parallel output plane using the 
    band-limited angular spectrum method.

    Args:
        field: A `tf.Tensor` of shape `(batchSize, params['upsample'] * pixelsX,
        params['upsample'] * pixelsY)` and dtype `tf.complex64` specifying the 
        input electric fields to be diffracted to the output plane.

        propagator: a `tf.Tensor` of shape `(batchSize, params['upsample'] * pixelsX,
        params['upsample'] * pixelsY)` and dtype `tf.complex64` defining the 
        reciprocal space, band-limited angular spectrum propagator.

        upsample: An odd-valued `int` specifying the factor by which the
        transverse field data stored in `field` should be upsampled.
    Returns:
        out: A `tf.Tensor` of shape `(batchSize, params['upsample'] * pixelsX,
        params['upsample'] * pixelsY)` and dtype `tf.complex64` specifying the 
        the electric fields at the output plane.
  '''

  # Zero pad `field` to be a stack of 2n-1 x 2n-1 matrices
  # Put batch parameter last for padding then transpose back.
  _, _, m = field.shape
  n = upsample * m
  field = tf.transpose(field, perm = [1, 2, 0])
  field_real = tf.math.real(field)
  field_imag = tf.math.imag(field)
  field_real = tf.image.resize(field_real, [n, n], method = 'nearest')
  field_imag = tf.image.resize(field_imag, [n, n], method = 'nearest')
  field = tf.cast(field_real, dtype = tf.complex64) + 1j * tf.cast(field_imag, dtype = tf.complex64)
  field = tf.image.resize_with_crop_or_pad(field, 2 * n - 1, 2 * n - 1)
  field = tf.transpose(field, perm = [2, 0, 1])

  # Apply the propagator in Fourier space.
  field_freq = tf.signal.fftshift(tf.signal.fft2d(field), axes = (1, 2))
  field_filtered = tf.signal.ifftshift(field_freq * propagator, axes = (1, 2))
  out = tf.signal.ifft2d(field_filtered)

  # Crop back down to n x n matrices.
  out = tf.transpose(out, perm = [1, 2, 0])
  out = tf.image.resize_with_crop_or_pad(out, n, n)
  out = tf.transpose(out, perm = [2, 0, 1])

  return out


def define_input_fields(params):
  '''
    Given the batch of input conditions with different wavelengths and incidence
    angles, this gives the input source fields incident on the metasurface.

    Args:
        params: A `dict` containing simulation and optimization settings.
    Returns:
        A `tf.Tensor` of shape `(batchSize, pixelsX, pixelsY)` and dtype 
        `tf.complex64` specifying the source fields injected onto a metasurface
        at the input.
  '''

  # Define the cartesian cross section.
  pixelsX = params['pixelsX']
  pixelsY = params['pixelsY']
  dx = params['Lx'] # grid resolution along x
  dy = params['Ly'] # grid resolution along y
  xa = np.linspace(0, pixelsX - 1, pixelsX) * dx # x axis array
  xa = xa - np.mean(xa) # center x axis at zero
  ya = np.linspace(0, pixelsY - 1, pixelsY) * dy # y axis vector
  ya = ya - np.mean(ya) # center y axis at zero
  [y_mesh, x_mesh] = np.meshgrid(ya, xa)
  x_mesh = x_mesh[np.newaxis, :, :]
  y_mesh = y_mesh[np.newaxis, :, :]

  # Extract the batch of wavelengths and input thetas.
  lam_phase_test = params['lam0'][:, 0, 0, 0, 0, 0]
  lam_phase_test = lam_phase_test[:, tf.newaxis, tf.newaxis]
  theta_phase_test = params['theta'][:, 0, 0, 0, 0, 0]
  theta_phase_test = theta_phase_test[:, tf.newaxis, tf.newaxis]

  # Apply a linear phase ramp based on the wavelength and thetas.
  phase_def = 2 * np.pi * np.sin(theta_phase_test) * x_mesh / lam_phase_test
  phase_def = tf.cast(phase_def, dtype = tf.complex64)

  return tf.exp(1j * phase_def)


def simulate(ER_t, UR_t, params = initialize_params()):
  '''
    Calculates the transmission/reflection coefficients for a unit cell with a
    given permittivity/permeability distribution and the batch of input conditions 
    (e.g., wavelengths, wavevectors, polarizations) for a fixed real space grid
    and number of Fourier harmonics.

    Args:
        ER_t: A `tf.Tensor` of shape `(batchSize, pixelsX, pixelsY, Nlayer, Nx, Ny)`
        and dtype `tf.complex64` specifying the relative permittivity distribution
        of the unit cell.

        UR_t: A `tf.Tensor` of shape `(batchSize, pixelsX, pixelsY, Nlayer, Nx, Ny)`
        and dtype `tf.complex64` specifying the relative permeability distribution
        of the unit cell.

        params: A `dict` containing simulation and optimization settings.
    Returns:
        outputs: A `dict` containing the keys {'rx', 'ry', 'rz', 'R', 'ref', 
        'tx', 'ty', 'tz', 'T', 'TRN'} corresponding to the computed reflection/tranmission
        coefficients and powers.
  '''

  # Extract commonly used parameters from the `params` dictionary.
  batchSize = params['batchSize']
  pixelsX = params['pixelsX']
  pixelsY = params['pixelsY']
  Nlay = params['Nlay']
  PQ = params['PQ']

  ### Step 3: Build convolution matrices for the permittivity and permeability ###
  ERC = rcwa_utils.convmat(ER_t, PQ[0], PQ[1])
  URC = rcwa_utils.convmat(UR_t, PQ[0], PQ[1])

  ### Step 4: Wave vector expansion ###
  I = np.eye(np.prod(PQ), dtype = complex)
  I = tf.convert_to_tensor(I, dtype = tf.complex64)
  I = I[tf.newaxis, tf.newaxis, tf.newaxis, tf.newaxis, :, :]
  I = tf.tile(I, multiples = (batchSize, pixelsX, pixelsY, Nlay, 1, 1))
  Z = np.zeros((np.prod(PQ), np.prod(PQ)), dtype = complex)
  Z = tf.convert_to_tensor(Z, dtype = tf.complex64)
  Z = Z[tf.newaxis, tf.newaxis, tf.newaxis, tf.newaxis, :, :]
  Z = tf.tile(Z, multiples = (batchSize, pixelsX, pixelsY, Nlay, 1, 1))
  n1 = np.sqrt(params['er1'])
  n2 = np.sqrt(params['er2'])

  k0 = tf.cast(2 * np.pi / params['lam0'], dtype = tf.complex64)
  kinc_x0 = tf.cast(n1 * tf.sin(params['theta']) * tf.cos(params['phi']), dtype = tf.complex64)
  kinc_y0 = tf.cast(n1 * tf.sin(params['theta']) * tf.sin(params['phi']), dtype = tf.complex64)
  kinc_z0 = tf.cast(n1 * tf.cos(params['theta']), dtype = tf.complex64)
  kinc_z0 = kinc_z0[:, :, :, 0, :, :]

  # Unit vectors
  T1 = np.transpose([2 * np.pi / params['Lx'], 0])
  T2 = np.transpose([0, 2 * np.pi / params['Ly']])
  p_max = np.floor(PQ[0] / 2.0)
  q_max = np.floor(PQ[1] / 2.0)
  p = tf.constant(np.linspace(-p_max, p_max, PQ[0]), dtype = tf.complex64) # indices along T1
  p = p[tf.newaxis, tf.newaxis, tf.newaxis, tf.newaxis, :, tf.newaxis]
  p = tf.tile(p, multiples = (1, pixelsX, pixelsY, Nlay, 1, 1))
  q = tf.constant(np.linspace(-q_max, q_max, PQ[1]), dtype = tf.complex64) # indices along T2
  q = q[tf.newaxis, tf.newaxis, tf.newaxis, tf.newaxis, tf.newaxis, :]
  q = tf.tile(q, multiples = (1, pixelsX, pixelsY, Nlay, 1, 1))

  # Build Kx and Ky matrices
  kx_zeros = tf.zeros(PQ[1], dtype = tf.complex64)
  kx_zeros = kx_zeros[tf.newaxis, tf.newaxis, tf.newaxis, tf.newaxis, tf.newaxis, :]
  ky_zeros = tf.zeros(PQ[0], dtype = tf.complex64)
  ky_zeros = ky_zeros[tf.newaxis, tf.newaxis, tf.newaxis, tf.newaxis, :, tf.newaxis]
  kx = kinc_x0 - 2 * np.pi * p / (k0 * params['Lx']) - kx_zeros
  ky = kinc_y0 - 2 * np.pi * q / (k0 * params['Ly']) - ky_zeros

  kx_T = tf.transpose(kx, perm = [0, 1, 2, 3, 5, 4])
  KX = tf.reshape(kx_T, shape = (batchSize, pixelsX, pixelsY, Nlay, np.prod(PQ)))
  KX = tf.linalg.diag(KX)

  ky_T = tf.transpose(ky, perm = [0, 1, 2, 3, 5, 4])
  KY = tf.reshape(ky_T, shape = (batchSize, pixelsX, pixelsY, Nlay, np.prod(PQ)))
  KY = tf.linalg.diag(KY)

  KZref = tf.linalg.matmul(tf.math.conj(params['ur1'] * I), tf.math.conj(params['er1'] * I))
  KZref = KZref - tf.linalg.matmul(KX, KX) - tf.linalg.matmul(KY, KY)
  KZref = tf.math.sqrt(KZref)
  KZref = -tf.math.conj(KZref)

  KZtrn = tf.linalg.matmul(tf.math.conj(params['ur2'] * I), tf.math.conj(params['er2'] * I))
  KZtrn = KZtrn - tf.linalg.matmul(KX, KX) - tf.linalg.matmul(KY, KY)
  KZtrn = tf.math.sqrt(KZtrn)
  KZtrn = tf.math.conj(KZtrn)

  ### Step 5: Free Space ###
  KZ = I - tf.linalg.matmul(KX, KX) - tf.linalg.matmul(KY, KY)
  KZ = tf.math.sqrt(KZ)
  KZ = tf.math.conj(KZ)

  Q_free_00 = tf.linalg.matmul(KX, KY)
  Q_free_01 = I - tf.linalg.matmul(KX, KX)
  Q_free_10 = tf.linalg.matmul(KY, KY) - I
  Q_free_11 = -tf.linalg.matmul(KY, KX)
  Q_free_row0 = tf.concat([Q_free_00, Q_free_01], axis = 5)
  Q_free_row1 = tf.concat([Q_free_10, Q_free_11], axis = 5)
  Q_free = tf.concat([Q_free_row0, Q_free_row1], axis = 4)

  W0_row0 = tf.concat([I, Z], axis = 5)
  W0_row1 = tf.concat([Z, I], axis = 5)
  W0 = tf.concat([W0_row0, W0_row1], axis = 4)

  LAM_free_row0 = tf.concat([1j * KZ, Z], axis = 5)
  LAM_free_row1 = tf.concat([Z, 1j * KZ], axis = 5)
  LAM_free = tf.concat([LAM_free_row0, LAM_free_row1], axis = 4)

  V0 = tf.linalg.matmul(Q_free, tf.linalg.inv(LAM_free))

  ### Step 6: Initialize Global Scattering Matrix ###
  SG = dict({})
  SG_S11 = tf.zeros(shape = (2 * np.prod(PQ), 2 * np.prod(PQ)), dtype = tf.complex64)
  SG['S11'] = tensor_utils.expand_and_tile_tf(SG_S11, batchSize, pixelsX, pixelsY)

  SG_S12 = tf.eye(num_rows = 2 * np.prod(PQ), dtype = tf.complex64)
  SG['S12'] = tensor_utils.expand_and_tile_tf(SG_S12, batchSize, pixelsX, pixelsY)

  SG_S21 = tf.eye(num_rows = 2 * np.prod(PQ), dtype = tf.complex64)
  SG['S21'] = tensor_utils.expand_and_tile_tf(SG_S21, batchSize, pixelsX, pixelsY)

  SG_S22 = tf.zeros(shape = (2 * np.prod(PQ), 2 * np.prod(PQ)), dtype = tf.complex64)
  SG['S22'] = tensor_utils.expand_and_tile_tf(SG_S22, batchSize, pixelsX, pixelsY)

  ### Step 7: Calculate eigenmodes ###

  # Build the eigenvalue problem.
  P_00 = tf.linalg.matmul(KX, tf.linalg.inv(ERC))
  P_00 = tf.linalg.matmul(P_00, KY)

  P_01 = tf.linalg.matmul(KX, tf.linalg.inv(ERC))
  P_01 = tf.linalg.matmul(P_01, KX)
  P_01 = URC - P_01

  P_10 = tf.linalg.matmul(KY, tf.linalg.inv(ERC))
  P_10 = tf.linalg.matmul(P_10, KY) - URC

  P_11 = tf.linalg.matmul(-KY, tf.linalg.inv(ERC))
  P_11 = tf.linalg.matmul(P_11, KX)

  P_row0 = tf.concat([P_00, P_01], axis = 5)
  P_row1 = tf.concat([P_10, P_11], axis = 5)
  P = tf.concat([P_row0, P_row1], axis = 4)

  Q_00 = tf.linalg.matmul(KX, tf.linalg.inv(URC))
  Q_00 = tf.linalg.matmul(Q_00, KY)

  Q_01 = tf.linalg.matmul(KX, tf.linalg.inv(URC))
  Q_01 = tf.linalg.matmul(Q_01, KX)
  Q_01 = ERC - Q_01

  Q_10 = tf.linalg.matmul(KY, tf.linalg.inv(URC))
  Q_10 = tf.linalg.matmul(Q_10, KY) - ERC

  Q_11 = tf.linalg.matmul(-KY, tf.linalg.inv(URC))
  Q_11 = tf.linalg.matmul(Q_11, KX)

  Q_row0 = tf.concat([Q_00, Q_01], axis = 5)
  Q_row1 = tf.concat([Q_10, Q_11], axis = 5)
  Q = tf.concat([Q_row0, Q_row1], axis = 4)

  # Compute eignmodes for the layers in each pixel for the whole batch.
  OMEGA_SQ = tf.linalg.matmul(P, Q)
  LAM, W = tensor_utils.eig_general(OMEGA_SQ)
  LAM = tf.sqrt(LAM)
  LAM = tf.linalg.diag(LAM)

  V = tf.linalg.matmul(Q, W)
  V = tf.linalg.matmul(V, tf.linalg.inv(LAM))

  # Scattering matrices for the layers in each pixel for the whole batch.
  W_inv = tf.linalg.inv(W)
  V_inv = tf.linalg.inv(V)
  A = tf.linalg.matmul(W_inv, W0) + tf.linalg.matmul(V_inv, V0)
  B = tf.linalg.matmul(W_inv, W0) - tf.linalg.matmul(V_inv, V0)
  X = tf.linalg.expm(-LAM * k0 * params['L'])

  S = dict({})
  A_inv = tf.linalg.inv(A)
  S11_left = tf.linalg.matmul(X, B)
  S11_left = tf.linalg.matmul(S11_left, A_inv)
  S11_left = tf.linalg.matmul(S11_left, X)
  S11_left = tf.linalg.matmul(S11_left, B)
  S11_left = A - S11_left
  S11_left = tf.linalg.inv(S11_left)

  S11_right = tf.linalg.matmul(X, B)
  S11_right = tf.linalg.matmul(S11_right, A_inv)
  S11_right = tf.linalg.matmul(S11_right, X)
  S11_right = tf.linalg.matmul(S11_right, A)
  S11_right = S11_right - B
  S['S11'] = tf.linalg.matmul(S11_left, S11_right)

  S12_right = tf.linalg.matmul(B, A_inv)
  S12_right = tf.linalg.matmul(S12_right, B)
  S12_right = A - S12_right
  S12_left = tf.linalg.matmul(S11_left, X)
  S['S12'] = tf.linalg.matmul(S12_left, S12_right)

  S['S21'] = S['S12']
  S['S22'] = S['S11']

  # Update the global scattering matrices.
  for l in range(Nlay):
    S_layer = dict({})
    S_layer['S11'] = S['S11'][:, :, :, l, :, :]
    S_layer['S12'] = S['S12'][:, :, :, l, :, :]
    S_layer['S21'] = S['S21'][:, :, :, l, :, :]
    S_layer['S22'] = S['S22'][:, :, :, l, :, :]
    SG = rcwa_utils.redheffer_star_product(SG, S_layer)

  ### Step 8: Reflection side ###
  # Eliminate layer dimension for tensors as they are unchanging on this dimension.
  KX = KX[:, :, :, 0, :, :]
  KY = KY[:, :, :, 0, :, :]
  KZref = KZref[:, :, :, 0, :, :]
  KZtrn = KZtrn[:, :, :, 0, :, :]
  Z = Z[:, :, :, 0, :, :]
  I = I[:, :, :, 0, :, :]
  W0 = W0[:, :, :, 0, :, :]
  V0 = V0[:, :, :, 0, :, :]

  Q_ref_00 = tf.linalg.matmul(KX, KY)
  Q_ref_01 = params['ur1'] * params['er1'] * I - tf.linalg.matmul(KX, KX)
  Q_ref_10 = tf.linalg.matmul(KY, KY) - params['ur1'] * params['er1'] * I
  Q_ref_11 = -tf.linalg.matmul(KY, KX)
  Q_ref_row0 = tf.concat([Q_ref_00, Q_ref_01], axis = 4)
  Q_ref_row1 = tf.concat([Q_ref_10, Q_ref_11], axis = 4)
  Q_ref = tf.concat([Q_ref_row0, Q_ref_row1], axis = 3)

  W_ref_row0 = tf.concat([I, Z], axis = 4)
  W_ref_row1 = tf.concat([Z, I], axis = 4)
  W_ref = tf.concat([W_ref_row0, W_ref_row1], axis = 3)

  LAM_ref_row0 = tf.concat([-1j * KZref, Z], axis = 4)
  LAM_ref_row1 = tf.concat([Z, -1j * KZref], axis = 4)
  LAM_ref = tf.concat([LAM_ref_row0, LAM_ref_row1], axis = 3)

  V_ref = tf.linalg.matmul(Q_ref, tf.linalg.inv(LAM_ref))

  W0_inv = tf.linalg.inv(W0)
  V0_inv = tf.linalg.inv(V0)
  A_ref = tf.linalg.matmul(W0_inv, W_ref) + tf.linalg.matmul(V0_inv, V_ref)
  A_ref_inv = tf.linalg.inv(A_ref)
  B_ref = tf.linalg.matmul(W0_inv, W_ref) - tf.linalg.matmul(V0_inv, V_ref)

  SR = dict({})
  SR['S11'] = tf.linalg.matmul(-A_ref_inv, B_ref)
  SR['S12'] = 2 * A_ref_inv
  SR_S21 = tf.linalg.matmul(B_ref, A_ref_inv)
  SR_S21 = tf.linalg.matmul(SR_S21, B_ref)
  SR['S21'] = 0.5 * (A_ref - SR_S21)
  SR['S22'] = tf.linalg.matmul(B_ref, A_ref_inv)

  ### Step 9: Transmission side ###
  Q_trn_00 = tf.linalg.matmul(KX, KY)
  Q_trn_01 = params['ur2'] * params['er2'] * I - tf.linalg.matmul(KX, KX)
  Q_trn_10 = tf.linalg.matmul(KY, KY) - params['ur2'] * params['er2'] * I
  Q_trn_11 = -tf.linalg.matmul(KY, KX)
  Q_trn_row0 = tf.concat([Q_trn_00, Q_trn_01], axis = 4)
  Q_trn_row1 = tf.concat([Q_trn_10, Q_trn_11], axis = 4)
  Q_trn = tf.concat([Q_trn_row0, Q_trn_row1], axis = 3)

  W_trn_row0 = tf.concat([I, Z], axis = 4)
  W_trn_row1 = tf.concat([Z, I], axis = 4)
  W_trn = tf.concat([W_trn_row0, W_trn_row1], axis = 3)

  LAM_trn_row0 = tf.concat([1j * KZtrn, Z], axis = 4)
  LAM_trn_row1 = tf.concat([Z, 1j * KZtrn], axis = 4)
  LAM_trn = tf.concat([LAM_trn_row0, LAM_trn_row1], axis = 3)

  V_trn = tf.linalg.matmul(Q_trn, tf.linalg.inv(LAM_trn))

  W0_inv = tf.linalg.inv(W0)
  V0_inv = tf.linalg.inv(V0)
  A_trn = tf.linalg.matmul(W0_inv, W_trn) + tf.linalg.matmul(V0_inv, V_trn)
  A_trn_inv = tf.linalg.inv(A_trn)
  B_trn = tf.linalg.matmul(W0_inv, W_trn) - tf.linalg.matmul(V0_inv, V_trn)

  ST = dict({})
  ST['S11'] = tf.linalg.matmul(B_trn, A_trn_inv)
  ST_S12 = tf.linalg.matmul(B_trn, A_trn_inv)
  ST_S12 = tf.linalg.matmul(ST_S12, B_trn)
  ST['S12'] = 0.5 * (A_trn - ST_S12)
  ST['S21'] = 2 * A_trn_inv
  ST['S22'] = tf.linalg.matmul(-A_trn_inv, B_trn)

  ### Step 10: Compute global scattering matrix ###
  SG = rcwa_utils.redheffer_star_product(SR, SG)
  SG = rcwa_utils.redheffer_star_product(SG, ST)

  ### Step 11: Compute source parameters ###

  # Compute mode coefficients of the source.
  delta = np.zeros((batchSize, pixelsX, pixelsY, np.prod(PQ)))
  delta[:, :, :, int(np.prod(PQ) / 2.0)] = 1

  # Incident wavevector.
  kinc_x0_pol = tf.math.real(kinc_x0[:, :, :, 0, 0])
  kinc_y0_pol = tf.math.real(kinc_y0[:, :, :, 0, 0])
  kinc_z0_pol = tf.math.real(kinc_z0[:, :, :, 0])
  kinc_pol = tf.concat([kinc_x0_pol, kinc_y0_pol, kinc_z0_pol], axis = 3)

  # Calculate TE and TM polarization unit vectors.
  firstPol = True
  for pol in range(batchSize):
    if (kinc_pol[pol, 0, 0, 0] == 0.0 and kinc_pol[pol, 0, 0, 1] == 0.0):
      ate_pol = np.zeros((1, pixelsX, pixelsY, 3))
      ate_pol[:, :, :, 1] = 1
      ate_pol = tf.convert_to_tensor(ate_pol, dtype = tf.float32)
    else:
      # Calculation of `ate` for oblique incidence.
      n_hat = np.zeros((1, pixelsX, pixelsY, 3))
      n_hat[:, :, :, 0] = 1
      n_hat = tf.convert_to_tensor(n_hat, dtype = tf.float32)
      kinc_pol_iter = kinc_pol[pol, :, :, :]
      kinc_pol_iter = kinc_pol_iter[tf.newaxis, :, :, :]
      ate_cross = tf.linalg.cross(n_hat, kinc_pol_iter)
      ate_pol =  ate_cross / tf.norm(ate_cross, axis = 3, keepdims = True)

    if firstPol:
      ate = ate_pol
      firstPol = False
    else:
      ate = tf.concat([ate, ate_pol], axis = 0)

  atm_cross = tf.linalg.cross(kinc_pol, ate)
  atm = atm_cross / tf.norm(atm_cross, axis = 3, keepdims = True)
  ate = tf.cast(ate, dtype = tf.complex64)
  atm = tf.cast(atm, dtype = tf.complex64)

  # Decompose the TE and TM polarization into x and y components.
  EP = params['pte'] * ate + params['ptm'] * atm
  EP_x = EP[:, :, :, 0]
  EP_x = EP_x[:, :, :, tf.newaxis]
  EP_y = EP[:, :, :, 1]
  EP_y = EP_y[:, :, :, tf.newaxis]

  esrc_x = EP_x * delta
  esrc_y = EP_y * delta
  esrc = tf.concat([esrc_x, esrc_y], axis = 3)
  esrc = esrc[:, :, :, :, tf.newaxis]

  W_ref_inv = tf.linalg.inv(W_ref)

  ### Step 12: Compute reflected and transmitted fields ###
  csrc = tf.linalg.matmul(W_ref_inv, esrc)

  # Compute tranmission and reflection mode coefficients.
  cref = tf.linalg.matmul(SG['S11'], csrc)
  ctrn = tf.linalg.matmul(SG['S21'], csrc)
  eref = tf.linalg.matmul(W_ref, cref)
  etrn = tf.linalg.matmul(W_trn, ctrn)

  rx = eref[:, :, :, 0 : np.prod(PQ), :]
  ry = eref[:, :, :, np.prod(PQ) : 2 * np.prod(PQ), :]
  tx = etrn[:, :, :, 0 : np.prod(PQ), :]
  ty = etrn[:, :, :, np.prod(PQ) : 2 * np.prod(PQ), :]

  # Compute longitudinal components.
  KZref_inv = tf.linalg.inv(KZref)
  KZtrn_inv = tf.linalg.inv(KZtrn)
  rz = tf.linalg.matmul(KX, rx) + tf.linalg.matmul(KY, ry)
  rz = tf.linalg.matmul(-KZref_inv, rz)
  tz = tf.linalg.matmul(KX, tx) + tf.linalg.matmul(KY, ty)
  tz = tf.linalg.matmul(-KZtrn_inv, tz)

  ### Step 13: Compute diffraction efficiences ###
  rx2 = tf.math.real(rx) ** 2 + tf.math.imag(rx) ** 2
  ry2 = tf.math.real(ry) ** 2 + tf.math.imag(ry) ** 2
  rz2 = tf.math.real(rz) ** 2 + tf.math.imag(rz) ** 2
  R2 = rx2 + ry2 + rz2
  R = tf.math.real(-KZref / params['ur1']) / tf.math.real(kinc_z0 / params['ur1'])
  R = tf.linalg.matmul(R, R2)
  R = tf.reshape(R, shape = (batchSize, pixelsX, pixelsY, PQ[0], PQ[1]))
  REF = tf.math.reduce_sum(R, axis = [3, 4])

  tx2 = tf.math.real(tx) ** 2 + tf.math.imag(tx) ** 2
  ty2 = tf.math.real(ty) ** 2 + tf.math.imag(ty) ** 2
  tz2 = tf.math.real(tz) ** 2 + tf.math.imag(tz) ** 2
  T2 = tx2 + ty2 + tz2
  T = tf.math.real(KZtrn / params['ur2']) / tf.math.real(kinc_z0 / params['ur2'])
  T = tf.linalg.matmul(T, T2)
  T = tf.reshape(T, shape = (batchSize, pixelsX, pixelsY, PQ[0], PQ[1]))
  TRN = tf.math.reduce_sum(T, axis = [3, 4])

  # Store the transmission/reflection coefficients and powers in a dictionary.
  outputs = dict({})
  outputs['rx'] = rx
  outputs['ry'] = ry
  outputs['rz'] = rz
  outputs['R'] = R
  outputs['REF'] = REF
  outputs['tx'] = tx
  outputs['ty'] = ty
  outputs['tz'] = tz
  outputs['T'] = T
  outputs['TRN'] = TRN

  return outputs
