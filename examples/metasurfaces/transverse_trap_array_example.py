"""
**Imports**
"""
import sys
#sys.path.insert(0, '/home/ubuntu/rcwa_stuff/rcwa_tf/src')

#print(sys.path)

import tensorflow as tf
import numpy as np
import rcwa_utils
import tensor_utils
import solver
import fab
#import matplotlib.pyplot as plt
#from google.colab import drive
#drive.mount('/content/drive')

"""**Loss Function Definition**"""

def transverse_trap():

  # Global parameters dictionary.
  global params, N, i, start, interval, weight_min, weight_max

  if (i + 1) >= start:
    weight = (weight_max - weight_min) / (N - start) * (i + 1 - start) + weight_min
    eps_r = fab.blur_unit_cell(var_eps, params)
    eps_r = fab.binary_push(eps_r, weight, params)
  else:
    eps_r = var_eps

  # Generate permittivity and permeability distributions.
  ER_t, UR_t = solver.generate_arbitrary_epsilon(eps_r, params)

  # Simulate the system.
  outputs = solver.simulate(ER_t, UR_t, params)
  field = outputs['ty'][:, :, :, np.prod(params['PQ']) // 2, 0]
  focal_plane = solver.propagate(params['input'] * field, params)
  lower = (params['pixelsX'] * params['upsample']) // 4
  upper = 3 * (params['pixelsX'] * params['upsample']) // 4
  f1 = tf.abs(focal_plane[0, lower, lower])
  f2 = tf.abs(focal_plane[0, lower, upper])
  f3 = tf.abs(focal_plane[0, upper, lower])
  f4 = tf.abs(focal_plane[0, upper, upper])
  f1 = f1[tf.newaxis]
  f2 = f2[tf.newaxis]
  f3 = f3[tf.newaxis]
  f4 = f4[tf.newaxis]
  traps = tf.concat([f1, f2, f3, f4], axis = 0)

  # Maximize the electric field magnitude at the desired focal spot.
  return -tf.math.reduce_min(traps)

def printProgressBar(i,max,postText,n_bar=100):
    j= i/max
    sys.stdout.write('\r')
    sys.stdout.write(f"[{'=' * int(n_bar * j):{n_bar}s}] {int(100 * j)}%  {postText}")
    sys.stdout.flush()

"""**Setup and Initialize Variables**"""

# Initialize global `params` dictionary storing optimization and simulation settings.
params = solver.initialize_params(wavelengths = [632.0],
                                  thetas = [0.0],
                                  phis = [0.0],
                                  pte = [1.0],
                                  ptm = [0.0],
                                  pixelsX = 31,
                                  pixelsY = 31,
                                  erd = 6.76,
                                  ers = 2.25,
                                  PQ = [5, 5],
                                  Lx = 443.1,
                                  Ly = 443.1,
                                  L = [632.0, 632.0],
                                  Nx = 128,
                                  eps_min = 1.0,
                                  eps_max = 6.76,
                                  blur_radius = 100.0)

params['f'] = 3 * params['pixelsX'] * params['Lx']
params['propagator'] = solver.make_propagator(params)
params['input'] = solver.define_input_fields(params)

# Initialize the epsilon disribution variable.
var_shape = (1, params['pixelsX'], params['pixelsY'], params['Nlay'] - 1, params['Nx'], params['Ny'])
np.random.seed(0)
eps_initial = params['eps_min'] + (params['eps_max'] - params['eps_min']) * np.random.rand(*var_shape)
var_eps = tf.Variable(eps_initial, dtype = tf.float32)

"""**Optimize**"""

# Number of optimization iterations.
i = 0
N = 4 #1000
start = 1
weight_min = 100.0
weight_max = 1000.0

# Define an optimizer and data to be stored.
opt = tf.keras.optimizers.Adam(learning_rate = 1E-3)
loss = [] #np.zeros(N + 1)

# Compute initial loss and duty cycle.
#loss[0] = transverse_trap().numpy()
loss = np.append(loss, transverse_trap().numpy())
print('Loss: ' + str(loss[0]))
print('\nOptimizing...')

# Optimize.
for i in range(N):
  opt.minimize(transverse_trap, var_list = [var_eps])
  #loss[i + 1] = transverse_trap().numpy()
  loss = np.append(loss, transverse_trap().numpy())
  printProgressBar(i, N - 1, 'Complete')

print('Loss: ' + str(loss[N]))

"""**Display Learning Curve**"""

#plt.plot(loss)
#plt.xlabel('Iterations')
#plt.ylabel('Loss')
#plt.xlim(0, N)
#plt.show()

"""**Calculate the Focal Plane Intensity of the Optimized Structure**"""

eps_r = fab.blur_unit_cell(var_eps, params)
eps_r = fab.threshold(eps_r, params)
ER_t, UR_t = solver.generate_arbitrary_epsilon(eps_r, params)
outputs = solver.simulate(ER_t, UR_t, params)
field = outputs['ty'][:, :, :, np.prod(params['PQ']) // 2, 0]
params['upsample'] = 1
params['propagator'] = solver.make_propagator(params)
params['input'] = solver.define_input_fields(params)
focal_plane = solver.propagate(field, params)
lower = (params['pixelsX'] * params['upsample']) // 4
upper = 3 * (params['pixelsX'] * params['upsample']) // 4
f1 = tf.abs(focal_plane[0, lower, lower])
f2 = tf.abs(focal_plane[0, lower, upper])
f3 = tf.abs(focal_plane[0, upper, lower])
f4 = tf.abs(focal_plane[0, upper, upper])
f1 = f1[tf.newaxis]
f2 = f2[tf.newaxis]
f3 = f3[tf.newaxis]
f4 = f4[tf.newaxis]
traps = tf.concat([f1, f2, f3, f4], axis = 0)

# Maximize the electric field magnitude at the desired focal spot.
loss_snapped = (-tf.math.reduce_min(traps)).numpy()
loss = np.append(loss, loss_snapped)
#plt.imshow(tf.abs(focal_plane[0, :, :]) ** 2)
#plt.colorbar()

eps_r_np = tf.abs(ER_t).numpy()
eps_r_np = eps_r_np[0, :, :, 0, :, :]

eps_r_st = np.zeros((31 * 128, 31 * 128))

for x in range(31):
  for y in range(31):
    eps_pixel = eps_r_np[x, y, :, :]
    eps_r_st[y * 128 : (y + 1) * 128, x * 128 : (x + 1) * 128] = eps_pixel

print(np.min(eps_r_st))
print(loss_snapped)

#eps_r_np = np.reshape(eps_r_np, newshape = (31, 31 * 128, 128))
#eps_r_np = np.reshape(eps_r_np, newshape = (31 * 128, 31 * 128))

#plt.figure(figsize = (10, 10))
#plt.imshow(eps_r_st, interpolation = None, cmap = 'gray')

with open('./topology_full_meta_loss.txt', 'w') as f:
  np.savetxt(f, loss)

with open('./topology_full_meta_eps_r.txt', 'w') as f:
  np.savetxt(f, eps_r_st)

