# -*- coding: utf-8 -*-
"""longitudinal_trap_array_binary_10_example.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/114H_6ok7xRWTF_0qgsG2C5CYO8jWG60b

**Imports**
"""

#gpu_info = !nvidia-smi
#gpu_info = '\n'.join(gpu_info)
#if gpu_info.find('failed') >= 0:
#  print('Select the Runtime > "Change runtime type" menu to enable a GPU accelerator, ')
#  print('and then re-execute this cell.')
#else:
#  print(gpu_info)

import tensorflow as tf
import numpy as np
import rcwa_utils
import tensor_utils
import solver
import fab
#import matplotlib.pyplot as plt
import sys

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
  index = (params['pixelsX'] * params['upsample']) // 2

  # Ascii W = 01010111 (Starting at 1.0*D, increments of 0.2*D)
  D = params['pixelsX'] * params['Lx']
  propagator1 = solver.make_propagator(params, f = 1.2* D)
  focal_plane1 = solver.propagate(params['input'] * field, propagator1, params['upsample'])
  f1 = tf.abs(focal_plane1[0, index, index])

  propagator2 = solver.make_propagator(params, f = 1.6 * D)
  focal_plane2 = solver.propagate(params['input'] * field, propagator2, params['upsample'])
  f2 = tf.abs(focal_plane2[0, index, index])

  propagator3 = solver.make_propagator(params, f = 2.0 * D)
  focal_plane3 = solver.propagate(params['input'] * field, propagator3, params['upsample'])
  f3 = tf.abs(focal_plane3[0, index, index])

  propagator4 = solver.make_propagator(params, f = 2.2 * D)
  focal_plane4 = solver.propagate(params['input'] * field, propagator4, params['upsample'])
  f4 = tf.abs(focal_plane4[0, index, index])

  propagator5 = solver.make_propagator(params, f = 2.4 * D)
  focal_plane5 = solver.propagate(params['input'] * field, propagator5, params['upsample'])
  f5 = tf.abs(focal_plane5[0, index, index])

  f1 = f1[tf.newaxis]
  f2 = f2[tf.newaxis]
  f3 = f3[tf.newaxis]
  f4 = f4[tf.newaxis]
  f5 = f5[tf.newaxis]
  traps = tf.concat([f1, f2, f3, f4, f5], axis = 0)

  # Maximize the electric field magnitude at the desired focal spot.
  return -tf.math.reduce_min(traps)

def rayleigh_sommerfeld(fields, p_obs):
  # Generate phase mask.
  #phase_mask = metasurface_phase_generator(phase, params)
  #fields = tf.math.exp(1j * tf.cast(phase_mask, dtype = tf.complex64)) # shape = [batch, params['N']]
  #fields = fields[:, :, tf.newaxis] # shape = [batch, params['N'], 1]

  # r_obs.shape = [z_obs_pts] (r, z)
  p_obs = p_obs[tf.newaxis, tf.newaxis, :] # p_obs.shape = [1, 1, z_obs_pts] (batch, r', z)
  r_src = params['r']
  p_src = r_src[tf.newaxis, :, tf.newaxis] # p_src.shape = [1, params['N'], 1] (batch, r', z)

  R_prime = tf.math.sqrt(p_src ** 2 + p_obs ** 2) # ASSUME always on axis calculation for observation points--> shape = [1, params['N'], z_obs_pts] (batch, r', z))

  # Cast to complex
  R_prime = tf.cast(R_prime, dtype = tf.complex64)
  #print(R_prime.shape)
  p_obs = tf.cast(p_obs, dtype = tf.complex64)
  k = tf.cast(2 * np.pi / params['lam'], dtype = tf.complex64)
  k = k[:, :, tf.newaxis]
  #print(k.shape)

  G = 1 / (2 * np.pi) * p_obs / R_prime * (1 - 1j * k * R_prime) * tf.math.exp(1j * k * R_prime) / R_prime ** 2 # shape = [batch, params['N'], z_obs_pts] (batch, r', z)
  #G = k / (1j * 2 * np.pi) * p_obs / R_prime * tf.math.exp(1j * k * R_prime) / R_prime # shape = [batch, params['N'], z_obs_pts] (batch, r', z)
  #print(G.shape)
  dr_prime = params['R'] / params['N']
  #dr_prime = R_prime / params['N']
  r_dr_theta = p_src * dr_prime * 2 * np.pi # shape = [1, params['N'], 1] (batch, r', z)
  #r_dr_theta = R_prime * dr_prime * 2 * np.pi # shape = [1, params['N'], 1] (batch, r', z)
  integrand = fields * G * r_dr_theta # shape = [batch, params['N'], z_obs_pts] (batch, r', z)
  #print(integrand.shape)

  return tf.math.reduce_sum(integrand, axis = 1, keepdims = False)# shape = [batch, z_obs_pts] (batch, z)

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
                                  pixelsX = 57, #31,
                                  pixelsY = 57, #31,
                                  erd = 6.76,
                                  ers = 2.25,
                                  PQ = [5, 5],
                                  Lx = 443.1,
                                  Ly = 443.1,
                                  L = [1000.0, 632.0],
                                  Nx = 128,
                                  eps_min = 1.0,
                                  eps_max = 6.76,
                                  blur_radius = 175.0)

params['f'] = 3 * params['pixelsX'] * params['Lx']
#params['propagator'] = solver.make_propagator(params)
params['input'] = solver.define_input_fields(params)

# Initialize the epsilon disribution variable.
var_shape = (1, params['pixelsX'], params['pixelsY'], params['Nlay'] - 1, params['Nx'], params['Ny'])
np.random.seed(0)
eps_initial = params['eps_min'] + (params['eps_max'] - params['eps_min']) * np.random.rand(*var_shape)
var_eps = tf.Variable(eps_initial, dtype = tf.float32)

"""**Optimize**"""

# Number of optimization iterations.
i = 0
N = 2
start = 1
weight_min = 100.0
weight_max = 10000.0

# Define an optimizer and data to be stored.
opt = tf.keras.optimizers.Adam(learning_rate = 2E-4)
loss = [] #np.zeros(N + 1)

# Compute initial loss and duty cycle.
#loss[0] = transverse_trap().numpy()
loss = np.append(loss, transverse_trap().numpy())
print('Loss: ' + str(loss[0]))
print('\nOptimizing...')

# Optimize.
for i in range(N):
  opt.minimize(transverse_trap, var_list = [var_eps])
  loss = np.append(loss, transverse_trap().numpy())
  printProgressBar(i, N - 1, 'Complete')

print('\nLoss: ' + str(loss[N]))

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
#params['propagator'] = solver.make_propagator(params)
params['input'] = solver.define_input_fields(params)
index = (params['pixelsX'] * params['upsample']) // 2

D = params['pixelsX'] * params['Lx']
propagator1 = solver.make_propagator(params, f = 1.2* D)
focal_plane1 = solver.propagate(params['input'] * field, propagator1, params['upsample'])
f1 = tf.abs(focal_plane1[0, index, index])

propagator2 = solver.make_propagator(params, f = 1.6 * D)
focal_plane2 = solver.propagate(params['input'] * field, propagator2, params['upsample'])
f2 = tf.abs(focal_plane2[0, index, index])

propagator3 = solver.make_propagator(params, f = 2.0 * D)
focal_plane3 = solver.propagate(params['input'] * field, propagator3, params['upsample'])
f3 = tf.abs(focal_plane3[0, index, index])

propagator4 = solver.make_propagator(params, f = 2.2 * D)
focal_plane4 = solver.propagate(params['input'] * field, propagator4, params['upsample'])
f4 = tf.abs(focal_plane4[0, index, index])

propagator5 = solver.make_propagator(params, f = 2.4 * D)
focal_plane5 = solver.propagate(params['input'] * field, propagator5, params['upsample'])
f5 = tf.abs(focal_plane5[0, index, index])

f1 = f1[tf.newaxis]
f2 = f2[tf.newaxis]
f3 = f3[tf.newaxis]
f4 = f4[tf.newaxis]
f5 = f5[tf.newaxis]
traps = tf.concat([f1, f2, f3, f4, f5], axis = 0)

# Maximize the electric field magnitude at the desired focal spot.
loss_snapped = (-tf.math.reduce_min(traps)).numpy()
#plt.imshow(tf.abs(focal_plane1[0, :, :]) ** 2)
#plt.colorbar()

#plt.imshow(tf.abs(focal_plane2[0, :, :]) ** 2)
#plt.colorbar()

#plt.imshow(tf.abs(focal_plane3[0, :, :]) ** 2)
#plt.colorbar()

#plt.imshow(tf.abs(focal_plane4[0, :, :]) ** 2)
#plt.colorbar()

#plt.imshow(tf.abs(focal_plane5[0, :, :]) ** 2)
#plt.colorbar()
'''
on_axis = []
points = 100
z_values = np.linspace(10E-6, 60E-6, points)
for z in z_values:
  propagator = solver.make_propagator(params, f = z)
  focal_plane = solver.propagate(params['input'] * field, propagator, params['upsample'])
  focus = tf.abs(focal_plane[0, index, index]) ** 2
  on_axis = np.append(on_axis, focus)



#propagator1 = solver.make_propagator(params, f = 20E-6)
#focal_plane1 = solver.propagate(params['input'] * field, propagator1, params['upsample'])
#f1 = tf.abs(focal_plane1[0, index, index])

plt.plot(z_values * 1E6, on_axis, 'k.')
plt.xlabel('z (microns)')
plt.ylabel('Intensity')
'''
eps_r_np = tf.abs(ER_t).numpy()
eps_r_np = eps_r_np[0, :, :, 0, :, :]

eps_r_st = np.zeros((params['pixelsX'] * params['Nx'], params['pixelsX'] * params['Nx']))

for x in range(params['pixelsX']):
  for y in range(params['pixelsX']):
    eps_pixel = eps_r_np[x, y, :, :]
    eps_r_st[y * params['Nx'] : (y + 1) * params['Nx'], x * params['Nx'] : (x + 1) * params['Nx']] = eps_pixel

#print(np.min(eps_r_st))
print(loss_snapped)

#eps_r_np = np.reshape(eps_r_np, newshape = (31, 31 * 128, 128))
#eps_r_np = np.reshape(eps_r_np, newshape = (31 * 128, 31 * 128))

#plt.figure(figsize = (10, 10))
#plt.imshow(eps_r_st, interpolation = None, cmap = 'gray')

with open('./topology_full_meta_long_loss_W_v1.txt', 'w') as f:
  np.savetxt(f, loss)

with open('./topology_full_meta_long_eps_r_W_v1.txt', 'w') as f:
  np.savetxt(f, eps_r_st)

