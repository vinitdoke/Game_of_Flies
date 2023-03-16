import numpy as np
import numba
import time
from force_profiles import all_force_functions
from state_parameters import initialise
from acceleration_updater import accelerator
from state_parameters import _plot_dist
import matplotlib.pyplot as plt

fac = 0.92

#@numba.njit
def integrate(
        mof: list,
        pos_x: np.ndarray,
        pos_y: np.ndarray,
        pos_z: np.ndarray,
        vel_x: np.ndarray,
        vel_y: np.ndarray,
        vel_z: np.ndarray,
        limits: tuple,
        r_max :float, # max distance at which particles interact
        n_type_array: np.ndarray,
        max_particles_per_type: int,

        acc_x: np.ndarray,
        acc_y: np.ndarray,
        acc_z: np.ndarray,
        timestep
):
    """
    Inputs :
    
    """
    accelerator(mof, pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, limits, r_max, n_type_array, max_particles_per_type, acc_x, acc_y, acc_z)

    pos_x += vel_x * 0.5 * timestep
    pos_y += vel_y * 0.5 * timestep
    #pos_z += vel_z * 0.5 * timestep

    vel_x += acc_x * timestep
    vel_y += acc_y * timestep
    #vel_z += acc_z * timestep

    vel_x *= fac
    vel_y *= fac
    #vel_z *= fac

    pos_x += vel_x * 0.5 * timestep
    pos_y += vel_y * 0.5 * timestep
    #pos_z += vel_z * 0.5 * timestep

    pos_x[pos_x < 0] += 100
    pos_x[pos_x > 100] -= 100

    pos_y[pos_y < 0] += 100
    pos_y[pos_y > 100] -= 100
    
    pass

n_type = 3
sample_input = np.random.randint(5, 10, (n_type, n_type, 4))
sample_input[:, :, 2] *= -1
sample_input[:, :, 0] = sample_input[:, :, 1] - 2

mof = all_force_functions("cluster_distance_input", *sample_input)

n_type_arr = np.array([500, 100, 100])

pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, interact_matrix, max_particles = initialise(n_type_arr)

vel_x = np.zeros_like(pos_x)
vel_y = np.zeros_like(pos_x)
vel_z = np.zeros_like(pos_x)


acc_x = np.zeros_like(vel_x)
acc_y = np.zeros_like(vel_x)
acc_z = np.zeros_like(vel_x)

output = []

plt.figure(figsize=(12, 12), dpi=80)
for i in range(10000):
    #out = np.vstack([pos_x, pos_y]).T
    #output.append(out)
    dt = np.sqrt(np.max(vel_x * vel_x + vel_y * vel_y))
    
    if dt > 1e-15:
        dt = 0.4 / dt
    else:
        dt = 0.1
    

    integrate(mof, pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, (100, 100), 10, n_type_arr, max_particles, acc_x, acc_y, acc_z, dt)
    if i % 10 == 0:
        _plot_dist(pos_x, pos_y, pos_z, max_particles, 3)

#np.savez("vid", result = output)