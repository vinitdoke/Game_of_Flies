import force_profiles
import numpy as np
import numba

print(force_profiles.wrap_clusters_force_distance)
#def interaction

def accelerator(
        matrix_of_functions: list,
        pos_x: np.ndarray,
        pos_y: np.ndarray,
        pos_z: np.ndarray,
        vel_x: np.ndarray,
        vel_y: np.ndarray,
        vel_z: np.ndarray,
        limits: tuple,
        n_type_array: np.ndarray,
        max_particles_per_type: int
):
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
        }
    """
    pass

print(type((2,100)))