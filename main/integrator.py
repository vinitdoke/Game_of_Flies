import numpy as np
from numba import njit, prange
from acceleration_updater import accelerator

fac = 0.9


@njit
def integrate(
        pos_x: np.ndarray,
        pos_y: np.ndarray,
        pos_z: np.ndarray,
        vel_x: np.ndarray,
        vel_y: np.ndarray,
        vel_z: np.ndarray,
        limits: np.ndarray,
        r_max :float, # max distance at which particles interact
        num_particles: np.int64,
        parameter_matrix: np.ndarray,
        particle_type_index_array: np.ndarray,

        acc_x: np.ndarray,
        acc_y: np.ndarray,
        acc_z: np.ndarray,
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