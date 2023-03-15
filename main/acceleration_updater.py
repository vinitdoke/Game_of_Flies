from force_profiles import wrap_clusters_force_distance
import numpy as np
import numba

#def interaction

@numba.njit
def accelerator(
        matrix_of_functions: list,
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
        acc_z: np.ndarray # to avoid allocating inside function
) -> np.ndarray:
    """
    INPUTS
    1. matrix_of_force_functions
    2. state_variable_dict : dict
        {
        "pos_x"  : array,
        "pos_y"  : array,
        "pos_z"  : array,
        "vel_x"  : array,
        "vel_y"  : array,
        "vel_z"  : array,
        "limits" : tuple (min, max),
        "n_type_array" : ndarray,
        "max_particles_per_type": int
        }
    """
    acc_x *= 0
    acc_y *= 0
    acc_z *= 0

    num_particle_types = n_type_array.shape[0]
    for i1 in range(num_particle_types):
        for j1 in range(n_type_array[i1]):
            idx_1 = i1 * max_particles_per_type + j1
            pos_x_1 = pos_x[idx_1]
            pos_y_1 = pos_y[idx_1]
            pos_z_1 = pos_z[idx_1]
            for i2 in range(num_particle_types):
                for j2 in range(n_type_array[i2]):
                    idx_2 = i2 * max_particles_per_type + j2
                    if idx_2 != idx_1:
                        pos_x_2 = pos_x[idx_2]
                        pos_y_2 = pos_y[idx_2]
                        pos_z_2 = pos_z[idx_2]

                        distance = (pos_x_1 - pos_x_2) * (pos_x_1 - pos_x_2) + (pos_y_1 - pos_y_2) * (pos_y_1 - pos_y_2) + (pos_z_1 - pos_z_2) * (pos_z_1 - pos_z_2)

                        if distance < r_max * r_max:





    pass

