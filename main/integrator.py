import numpy as np
from numba import njit
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
        num_particles: int,
        parameter_matrix: np.ndarray,
        particle_type_index_array: np.ndarray,

        acc_x: np.ndarray,
        acc_y: np.ndarray,
        acc_z: np.ndarray,
        timestep: float = 0.1
) -> None:
    '''vel_x += acc_x * 0.5 * timestep
    vel_y += acc_y * 0.5 * timestep

    accelerator(pos_x, pos_y, pos_z, vel_x, vel_y, vel_z,
                limits, r_max, num_particles,
                parameter_matrix, particle_type_index_array,
                acc_x, acc_y, acc_z)
    
    pos_x += (vel_x + 0.5 * acc_x * timestep) * timestep
    pos_y += (vel_y + 0.5 * acc_y * timestep) * timestep

    vel_x += acc_x * 0.5 * timestep
    vel_y += acc_y * 0.5 * timestep

    vel_x *= fac
    vel_y *= fac'''
    accelerator(pos_x, pos_y, pos_z, vel_x, vel_y, vel_z,
                limits, r_max, num_particles,
                parameter_matrix, particle_type_index_array,
                acc_x, acc_y, acc_z)

    pos_x += vel_x * 0.5 * timestep
    pos_y += vel_y * 0.5 * timestep
    # pos_z += vel_z * 0.5 * timestep

    vel_x += acc_x * timestep
    vel_y += acc_y * timestep
    # vel_z += acc_z * timestep

    vel_x *= fac
    vel_y *= fac
    # vel_z *= fac

    pos_x += vel_x * 0.5 * timestep
    pos_y += vel_y * 0.5 * timestep
    # pos_z += vel_z * 0.5 * timestep

    pos_x[pos_x < 0] += limits[0]
    pos_x[pos_x > limits[0]] -= limits[0]
    pos_y[pos_y < 0] += limits[1]
    pos_y[pos_y > limits[1]] -= limits[1]