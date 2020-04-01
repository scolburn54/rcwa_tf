import tensorflow as tf
import numpy as np
import rcwa_utils
import tensor_utils
import solver
import time

def total_transmitted_power():

  # Generate permitivitty and permeability distributions.
  global params
  ER_t, UR_t = solver.generate_cylindrical_nanoposts(var_duty, params)

  # Simulate the system.
  outputs = solver.simulate(ER_t, UR_t, params)

  return (1 - outputs['REF'][0, 0, 0])

# Initialize duty cycle variable and global params dictionary.
params = solver.initialize_params()
params['erd'] = 6.76
params['ers'] = 2.25
params['PQ'] = [11, 11]
var_shape = (1, params['pixelsX'], params['pixelsY'])
duty_initial = 0.6 * np.ones(shape = var_shape)
var_duty = tf.Variable(duty_initial, dtype = tf.float32)

# Compute the initial, unoptimized permittivity distribution.
epsilon_r_initial, _ = solver.generate_cylindrical_nanoposts(var_duty, params)

# Number of optimization iterations.
N = 49

# Define an optimizer and data to be stored.
opt = tf.keras.optimizers.Adam(learning_rate = 1E-3)
loss = np.zeros(N + 1)
duty = np.zeros(N + 1)

# Compute initial loss and duty cycle.
duty[0] = tf.clip_by_value(var_duty, params['duty_min'], params['duty_max']).numpy()
loss[0] = total_transmitted_power().numpy()
print('Initial Duty Cycle: ' + str(duty[0]))
print('TRN: ' + str(loss[0]))
print('\nOptimizing...')

# Optimize.
prev_ten_percent = 0
curr_ten_percent = 0
t = time.time()
for i in range(N):
  opt.minimize(total_transmitted_power, var_list = [var_duty])
  loss[i + 1] = total_transmitted_power().numpy()
  duty[i + 1] = tf.clip_by_value(var_duty, params['duty_min'], params['duty_max']).numpy()
  prev_ten_percent = curr_ten_percent
  curr_ten_percent = np.int(10.0 * i / N)
  if (curr_ten_percent != prev_ten_percent):
      print(str(10 * curr_ten_percent) + '% complete')

print('\nFinal Duty Cycle: ' + str(duty[N]))
print('TRN: ' + str(loss[N]))

elapsed_time = time.time() - t
print('\nElapsed Time: ' + str(elapsed_time))

np.savetxt('loss.txt', loss)
np.savetxt('duty.txt', duty)
