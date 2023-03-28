import numpy as np
<<<<<<< HEAD
import numba
import time
from force_profiles import all_force_functions
from state_parameters import initialise
from acceleration_updater import accelerator
from state_parameters import _plot_dist

#@numba.njit
def integrate(
        mof: list,
=======
from numba import njit, prange
from acceleration_updater import accelerator

fac = 0.9


@njit
def integrate(
>>>>>>> object_oriented
        pos_x: np.ndarray,
        pos_y: np.ndarray,
        pos_z: np.ndarray,
        vel_x: np.ndarray,
        vel_y: np.ndarray,
        vel_z: np.ndarray,
<<<<<<< HEAD
        limits: tuple,
        r_max :float, # max distance at which particles interact
        n_type_array: np.ndarray,
        max_particles_per_type: int,
=======
        limits: np.ndarray,
        r_max :float, # max distance at which particles interact
        num_particles: np.int64,
        parameter_matrix: np.ndarray,
        particle_type_index_array: np.ndarray,
>>>>>>> object_oriented

        acc_x: np.ndarray,
        acc_y: np.ndarray,
        acc_z: np.ndarray,
<<<<<<< HEAD
        timestep
):
    """
    Inputs :
    
    """
    accelerator(mof, pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, limits, r_max, n_type_array, max_particles_per_type, acc_x, acc_y, acc_z)

    pos_x += vel_x * 0.5 * timestep
    pos_y += vel_y * 0.5 * timestep
    pos_z += vel_z * 0.5 * timestep

    vel_x += acc_x * timestep
    vel_y += acc_y * timestep
    vel_z += acc_z * timestep

    pos_x += vel_x * 0.5 * timestep
    pos_y += vel_y * 0.5 * timestep
    pos_z += vel_z * 0.5 * timestep
    
    pass

n_type = 3
sample_input = np.random.randint(1, 10, (n_type, n_type, 4))
sample_input[:, :, 2] *= -1
sample_input[:, :, 0] = sample_input[:, :, 1] - 2

mof = all_force_functions("cluster_distance_input", *sample_input)

n_type_arr = np.array([10, 10, 10])

pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, interact_matrix, max_particles = initialise(n_type_arr)

acc_x = np.zeros_like(vel_x)
acc_y = np.zeros_like(vel_x)
acc_z = np.zeros_like(vel_x)

output = []

for i in range(2):
    out = np.vstack([pos_x, pos_y]).T
    output.append(out)
    integrate(mof, pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, (100, 100), 10, n_type_arr, max_particles, acc_x, acc_y, acc_z, 2)
    #_plot_dist(pos_x, pos_y, pos_z, max_particles, 3)

np.savez("vid", result = output)
=======
        timestep: np.float64 = None
) -> None:
    
    if timestep is None:
        timestep = 0
        for i in prange(num_particles):
            tmp = vel_x[i] * vel_x[i] + vel_y[i] * vel_y[i] + \
                vel_z[i] * vel_z[i]
            if tmp > timestep:
                timestep = tmp
        timestep = np.sqrt(timestep)
        
        if timestep > 1e-15:
            timestep = 0.2 / timestep
        else:
            timestep = 0.1
    
    
    for i in prange(num_particles):
        vel_x[i] += acc_x[i] * 0.5 * timestep
        vel_y[i] += acc_y[i] * 0.5 * timestep
        vel_z[i] += acc_z[i] * 0.5 * timestep

    accelerator(pos_x, pos_y, pos_z, vel_x, vel_y, vel_z,
                limits, r_max, num_particles,
                parameter_matrix, particle_type_index_array,
                acc_x, acc_y, acc_z)
    
    for i in prange(num_particles):
        pos_x[i] += (vel_x[i] + 0.5 * acc_x[i] * timestep) * timestep
        pos_y[i] += (vel_y[i] + 0.5 * acc_y[i] * timestep) * timestep
        pos_z[i] += (vel_z[i] + 0.5 * acc_z[i] * timestep) * timestep

        vel_x[i] += acc_x[i] * 0.5 * timestep
        vel_y[i] += acc_y[i] * 0.5 * timestep
        vel_z[i] += acc_z[i] * 0.5 * timestep

        vel_x[i] *= fac
        vel_y[i] *= fac
        vel_z[i] *= fac

        if pos_x[i] < 0:
            pos_x[i] += limits[0]
        elif pos_x[i] > limits[0]:
            pos_x[i] -= limits[0]
        if pos_y[i] < 0:
            pos_y[i] += limits[1]
        elif pos_y[i] > limits[1]:
            pos_y[i] -= limits[1]
        if pos_z[i] < 0:
            pos_z[i] += limits[2]
        elif pos_z[i] > limits[2]:
            pos_z[i] -= limits[2]
>>>>>>> object_oriented
