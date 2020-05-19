import tensorflow as tf
import numpy as np
import rcwa_utils
import tensor_utils
import solver
import time

def focal_spot():

  # Generate permitivitty and permeability distributions.
  global params
  ER_t, UR_t = solver.generate_coupled_cylindrical_resonators(r_x_var, r_y_var, params)

  # Simulate the system.
  outputs = solver.simulate(ER_t, UR_t, params)
  field = outputs['ty'][:, :, :, np.prod(params['PQ']) // 2, 0]
  focal_plane = solver.propagate(params['input'] * field, params)
  index = (params['pixelsX'] * params['upsample']) // 2
  f1 = tf.abs(focal_plane[0, index, index])
  f2 = tf.abs(focal_plane[1, index, index])
  loss = -f1 * f2

  return loss

# Initialize global params dictionary and overwrite default values.
setup_t0 = time.time()
params = solver.initialize_params()
params['batchSize'] = 2
params['erd'] = 6.76
params['ers'] = 2.25
params['PQ'] = [5, 5]
params['f'] = 30E-6
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
params['upsample'] = 11
params['Lx'] = 0.7 * 632 * params['nanometers'] # period along x
params['Ly'] = params['Lx'] # period along y
length_shape = (1, 1, 1, params['Nlay'], 1, 1)
params['L'] = 632 * params['nanometers'] * tf.ones(shape = length_shape, dtype = tf.complex64)

# Define the batch parameters and duty cycle variable.
simulation_shape = (batchSize, pixelsX, pixelsY)
batch_shape = (batchSize, pixelsX, pixelsY, 1, 1, 1)
pol_shape = (batchSize, pixelsX, pixelsY, 1)
lam0 = params['nanometers'] * tf.convert_to_tensor([632.0, 530.0], dtype = tf.float32)
lam0 = lam0[:, tf.newaxis, tf.newaxis, tf.newaxis, tf.newaxis, tf.newaxis]
lam0 = tf.tile(lam0, multiples = (1, pixelsX, pixelsY, 1, 1, 1))
params['lam0'] = lam0

theta = params['degrees'] * tf.convert_to_tensor([0.0, 0.0], dtype = tf.float32)
theta = theta[:, tf.newaxis, tf.newaxis, tf.newaxis, tf.newaxis, tf.newaxis]
theta = tf.tile(theta, multiples = (1, pixelsX, pixelsY, 1, 1, 1))

params['theta'] = theta
params['phi'] = 0 * params['degrees'] * tf.ones(shape = batch_shape, dtype = tf.float32)
params['pte'] = 1 * tf.ones(shape = pol_shape, dtype = tf.complex64)
params['ptm'] = 0 * tf.ones(shape = pol_shape, dtype = tf.complex64)
params['propagator'] = solver.make_propagator(params)
params['input'] = solver.define_input_fields(params)

var_shape = (1, pixelsX, pixelsY, 4)
r_x_initial = 0.175 * np.ones(shape = var_shape)
r_y_initial = r_x_initial
r_x_var = tf.Variable(r_x_initial, dtype = tf.float32)
r_y_var = tf.Variable(r_y_initial, dtype = tf.float32)

# Compute the initial, unoptimized permittivity distribution.
epsilon_r_initial, mu_r_initial = solver.generate_coupled_cylindrical_resonators(r_x_var, r_y_var, params)

# Number of optimization iterations.
N = 152

# Define an optimizer and data to be stored.
opt = tf.keras.optimizers.Adam(learning_rate = 5E-4)
loss = np.zeros(N + 1)
duty = np.zeros(N + 1)
setup_time = time.time() - setup_t0

# Compute initial loss and duty cycle.
loss[0] = focal_spot().numpy()
print('Loss: ' + str(loss[0]))
print('\nOptimizing...')

# Optimize.
prev_ten_percent = 0
curr_ten_percent = 0
t = time.time()
for i in range(N):
  opt.minimize(focal_spot, var_list = [r_x_var, r_y_var])
  loss[i + 1] = focal_spot().numpy()
  prev_ten_percent = curr_ten_percent
  curr_ten_percent = np.int(10.0 * i / N)
  if (curr_ten_percent != prev_ten_percent):
      print(str(10 * curr_ten_percent) + '% complete')

elapsed_time = time.time() - t
print('Setup Time: ' + str(setup_time))
print('Elapsed Time: ' + str(elapsed_time))
print('Loss: ' + str(loss[N]))

# Simulate the system.
outputs = solver.simulate(epsilon_r_initial, mu_r_initial, params)
field_initial = outputs['ty'][:, :, :, np.prod(params['PQ']) // 2, 0]
focal_plane_initial = solver.propagate(field_initial, params)

ER_t, UR_t = solver.generate_coupled_cylindrical_resonators(r_x_var, r_y_var, params)

# Simulate the system.
outputs = solver.simulate(ER_t, UR_t, params)
field = outputs['ty'][:, :, :, np.prod(params['PQ']) // 2, 0]
focal_plane = solver.propagate(field, params)

# Save data
np.savetxt('loss.txt', loss)
np.save('focal_plane_initial.npy', focal_plane_initial)
np.save('focal_plane_opt.npy', focal_plane)
np.save('field_initial.npy', field_initial)
np.save('field_opt.npy', field)
np.save('r_x_initial.npy', r_x_initial)
np.save('r_y_initial.npy', r_y_initial)
np.save('r_x_final.npy', r_x_var.numpy())
np.save('r_y_final.npy', r_y_var.numpy())
np.save('epsilon_r_initial', epsilon_r_initial)
np.save('epsilon_r_final', ER_t)
