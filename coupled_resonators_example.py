import tensorflow as tf
import numpy as np
import rcwa_utils
import tensor_utils
import solver

def total_reflected_power():

  # Generate permitivitty and permeability distributions.
  global params
  ER_t, UR_t = solver.generate_arbitrary_epsilon(eps_var, params)

  # Simulate the system.
  outputs = solver.simulate(ER_t, UR_t, params)

  # Calculate the loss.
  global desired_amp, desired_phase, actual_complex
  actual_complex = outputs['ty'][:, 0, 0, np.prod(params['PQ']) // 2, 0]
  diff = actual_complex - desired_amp * tf.exp(1j * desired_phase)
  #diff = tf.exp(1j * tf.cast(tf.math.angle(actual_complex), dtype = tf.complex64)) - tf.exp(1j * desired_phase)
  loss = tf.reduce_sum(tf.square(tf.abs(diff)))

  return loss

def focal_spot():

  # Generate permitivitty and permeability distributions.
  global params
  #ER_t, UR_t = solver.generate_arbitrary_epsilon(eps_var, params)
  ER_t, UR_t = generate_coupled_cylindrical_resonators(r_x_var, r_y_var, params)

  # Simulate the system.
  outputs = solver.simulate(ER_t, UR_t, params)
  field = outputs['ty'][:, :, :, np.prod(params['PQ']) // 2, 0]
  focal_plane = solver.propagate(params['input'] * field, params)
  f1 = tf.abs(focal_plane[0, params['pixelsX'] // 2, params['pixelsX'] // 2])
  #f2 = tf.abs(focal_plane[1, params['pixelsX'] // 2, params['pixelsX'] // 2])
  #f3 = tf.abs(focal_plane[2, params['pixelsX'] // 2, params['pixelsX'] // 2])
  #f4 = tf.abs(focal_plane[3, params['pixelsX'] // 2, params['pixelsX'] // 2])
  #loss = -f1 * f2 * f3 * f4
  loss = -f1

  return loss

def generate_coupled_cylindrical_resonators(r_x, r_y, params):

  # Retrieve simulation size parameters.
  batchSize = params['batchSize']
  pixelsX = params['pixelsX']
  pixelsY = params['pixelsY']
  Nlay = params['Nlay']
  Nx = params['Nx']
  Ny = params['Ny']
  Lx = params['Lx']
  Ly = params['Ly']

  # Initialize relative permeability
  materials_shape = (batchSize, pixelsX, pixelsY, Nlay, Nx, Ny)
  UR = params['urd'] * np.ones(materials_shape)

  # Define the cartesian cross section
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

  c1_x = -Lx / 4
  c1_y = -Ly / 4
  c2_x = -Lx / 4
  c2_y = Ly / 4
  c3_x = Lx / 4
  c3_y = -Ly / 4
  c4_x = Lx / 4
  c4_y = Ly / 4

  r_x = params['Lx'] * tf.clip_by_value(r_x, clip_value_min = 0.01, clip_value_max = 0.25)
  r_y = params['Ly'] * tf.clip_by_value(r_y, clip_value_min = 0.01, clip_value_max = 0.25)
  r_x = r_x[:, :, :, tf.newaxis, tf.newaxis, tf.newaxis, :]
  r_y = r_y[:, :, :, tf.newaxis, tf.newaxis, tf.newaxis, :]

  c1 = 1 - ((x_mesh - c1_x) / r_x[:, :, :, :, :, :, 0]) ** 2 - ((y_mesh - c1_y) / r_y[:, :, :, :, :, :, 0]) ** 2
  c2 = 1 - ((x_mesh - c2_x) / r_x[:, :, :, :, :, :, 1]) ** 2 - ((y_mesh - c2_y) / r_y[:, :, :, :, :, :, 1]) ** 2
  c3 = 1 - ((x_mesh - c3_x) / r_x[:, :, :, :, :, :, 2]) ** 2 - ((y_mesh - c3_y) / r_y[:, :, :, :, :, :, 2]) ** 2
  c4 = 1 - ((x_mesh - c4_x) / r_x[:, :, :, :, :, :, 3]) ** 2 - ((y_mesh - c4_y) / r_y[:, :, :, :, :, :, 3]) ** 2

  # Build device layer
  ER_c1 = tf.math.sigmoid(params['sigmoid_coeff'] * c1)
  ER_c2 = tf.math.sigmoid(params['sigmoid_coeff'] * c2)
  ER_c3 = tf.math.sigmoid(params['sigmoid_coeff'] * c3)
  ER_c4 = tf.math.sigmoid(params['sigmoid_coeff'] * c4)
  ER_t = 1 + (params['erd'] - 1) * (ER_c1 + ER_c2 + ER_c3 + ER_c4)

  # Build substrate and concatenate along the layers dimension
  device_shape = (batchSize, pixelsX, pixelsY, 1, Nx, Ny)
  ER_substrate = params['ers'] * tf.ones(device_shape, dtype = tf.float32)
  ER_t = tf.concat(values = [ER_t, ER_substrate], axis = 3)

  # Cast to complex for subsequent calculations
  ER_t = tf.cast(ER_t, dtype = tf.complex64)
  UR_t = tf.convert_to_tensor(UR, dtype = tf.float32)
  UR_t = tf.cast(UR_t, dtype = tf.complex64)

  return ER_t, UR_t

# Initialize global params dictionary and overwrite default values.
params = solver.initialize_params()
params['batchSize'] = 1
params['erd'] = 4.0
params['ers'] = 2.25
params['PQ'] = [3, 3]
params['f'] = 15E-6
batchSize = params['batchSize']
num_pixels = 31
pixelsX = num_pixels
pixelsY = num_pixels
params['pixelsX'] = pixelsX
params['pixelsY'] = pixelsY
Nlay = params['Nlay']
Nx = 128
params['Nx'] = Nx
Ny = 128
params['Ny'] = Ny
params['sigmoid_coeff'] = 1000.0

# Define the batch parameters and duty cycle variable.
simulation_shape = (batchSize, pixelsX, pixelsY)
batch_shape = (batchSize, pixelsX, pixelsY, 1, 1, 1)
pol_shape = (batchSize, pixelsX, pixelsY, 1)
lam0 = params['nanometers'] * tf.convert_to_tensor([633.0], dtype = tf.float32)
lam0 = lam0[:, tf.newaxis, tf.newaxis, tf.newaxis, tf.newaxis, tf.newaxis]
lam0 = tf.tile(lam0, multiples = (1, pixelsX, pixelsY, 1, 1, 1))
params['lam0'] = lam0
theta = params['degrees'] * tf.convert_to_tensor([0.0], dtype = tf.float32)
theta = theta[:, tf.newaxis, tf.newaxis, tf.newaxis, tf.newaxis, tf.newaxis]
theta = tf.tile(theta, multiples = (1, pixelsX, pixelsY, 1, 1, 1))
params['theta'] = theta
params['phi'] = 0 * params['degrees'] * tf.ones(shape = batch_shape, dtype = tf.float32)
params['pte'] = 1 * tf.ones(shape = pol_shape, dtype = tf.complex64)
params['ptm'] = 0 * tf.ones(shape = pol_shape, dtype = tf.complex64)
params['propagator'] = solver.make_propagator(params)
params['input'] = solver.define_input_fields(params)

var_shape = (batchSize, pixelsX, pixelsY, 4)
np.random.RandomState(seed = 0)
r_x_initial = np.random.normal(loc = 0.125, scale = 0.025, size = var_shape)
r_y_initial = np.random.normal(loc = 0.125, scale = 0.025, size = var_shape)
r_x_var = tf.Variable(r_x_initial, dtype = tf.float32)
r_y_var = tf.Variable(r_y_initial, dtype = tf.float32)

# Compute the initial, unoptimized permittivity distribution.
epsilon_r_initial, mu_r_initial = generate_coupled_cylindrical_resonators(r_x_var, r_y_var, params)

# Number of optimization iterations.
N = 25

# Define an optimizer and data to be stored.
opt = tf.keras.optimizers.Adam(learning_rate = 0.1)
loss = np.zeros(N + 1)
duty = np.zeros(N + 1)

# Compute initial loss and duty cycle.
loss[0] = focal_spot().numpy()
print('Loss: ' + str(loss[0]))
print('\nOptimizing...')

# Optimize.
for i in range(N):
  opt.minimize(focal_spot, var_list = [r_x_var, r_y_var])
  loss[i + 1] = focal_spot().numpy()

print('Loss: ' + str(loss[N]))

plt.imshow(np.abs(epsilon_r_initial[0, 1, 0, 0, :, :]))
plt.colorbar()
plt.title('Initial Unit Cell Permittivity')

epsilon_r_final, _ = generate_coupled_cylindrical_resonators(r_x_var, r_y_var, params)

# Simulate the system.
outputs = solver.simulate(epsilon_r_initial, mu_r_initial, params)
field = outputs['ty'][:, :, :, np.prod(params['PQ']) // 2, 0]
focal_plane = solver.propagate(field, params)
plt.imshow(np.abs(focal_plane[0, :, :]) ** 2)
plt.colorbar()

ER_t, UR_t = generate_coupled_cylindrical_resonators(r_x_var, r_y_var, params)

# Simulate the system.
outputs = solver.simulate(ER_t, UR_t, params)
field = outputs['ty'][:, :, :, np.prod(params['PQ']) // 2, 0]
focal_plane = solver.propagate(field, params)
