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
        timestep: float = None
) -> None:
    if timestep is None:
        timestep = np.sqrt(np.max(vel_x[:num_particles] * vel_x[:num_particles] + vel_y[:num_particles] * vel_y[:num_particles]))
        
        if timestep > 1e-15:
            timestep = 0.2 / timestep
        else:
            timestep = 0.1
    
    vel_x += acc_x * 0.5 * timestep
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
    vel_y *= fac

    pos_x[pos_x < 0] += limits[0]
    pos_x[pos_x > limits[0]] -= limits[0]
    pos_y[pos_y < 0] += limits[1]
    pos_y[pos_y > limits[1]] -= limits[1]