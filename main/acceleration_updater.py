from force_profiles import all_force_functions
from state_parameters import initialise
from state_parameters import _plot_dist
import numpy as np
import numba
import time

# TODO: Apply periodic boundary condition

lim = 0
a = 5


@numba.njit
def resolve_particle_group(
        force_function,
        index_1: int,  # the indices of the particle types to be resolved
        index_2: int,
        pos_x: np.ndarray,
        pos_y: np.ndarray,
        pos_z: np.ndarray,
        vel_x: np.ndarray,
        vel_y: np.ndarray,
        vel_z: np.ndarray,
        limits: tuple,
        r_max: float,  # max distance at which particles interact
        n_type_array: np.ndarray,
        max_particles_per_type: int,

        acc_x: np.ndarray,
        acc_y: np.ndarray,
        acc_z: np.ndarray  # to avoid allocating inside function
) -> None:
    for i1 in range(n_type_array[index_1]):
        p1 = max_particles_per_type * index_1 + i1
        pos_x_1 = pos_x[p1]
        pos_y_1 = pos_y[p1]
        for i2 in range(n_type_array[index_2]):
            p2 = max_particles_per_type * index_2 + i2
            if i1 != i2:
                pos_x_2 = pos_x[p2]
                pos_y_2 = pos_y[p2]

                distance = (pos_x_1 - pos_x_2) * (pos_x_1 - pos_x_2) + (
                            pos_y_1 - pos_y_2) * (pos_y_1 - pos_y_2)

                if 1e-10 < distance < r_max * r_max:  # 1e-10 is r_min**2
                    distance = np.sqrt(distance)
                    acc = force_function(distance)
                    a_x = acc * (pos_x_1 - pos_x_2) / distance
                    a_y = acc * (pos_y_1 - pos_y_2) / distance

                    acc_x[p1] -= a_x
                    acc_y[p1] -= a_y

                    if pos_x_1 < lim:
                        acc_x[p1] += a
                    elif pos_x_1 > 100 - lim:
                        acc_x[p1] -= a

                    if pos_y_1 < lim:
                        acc_y[p1] += a
                    elif pos_y_1 > 100 - lim:
                        acc_y[p1] -= a


def accelerator(
        matrix_of_functions: list,
        pos_x: np.ndarray,
        pos_y: np.ndarray,
        pos_z: np.ndarray,
        vel_x: np.ndarray,
        vel_y: np.ndarray,
        vel_z: np.ndarray,
        limits: tuple,
        r_max: float,  # max distance at which particles interact
        n_type_array: np.ndarray,
        max_particles_per_type: int,

        acc_x: np.ndarray,
        acc_y: np.ndarray,
        acc_z: np.ndarray  # to avoid allocating inside function
) -> None:
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
    for j1 in range(num_particle_types):
        for j2 in range(num_particle_types):
            force_function = matrix_of_functions[j1][j2]
            resolve_particle_group(force_function, j1, j2, pos_x, pos_y, pos_z,
                                   vel_x, vel_y, vel_z, (100, 100), 10,
                                   n_type_array, max_particles_per_type, acc_x,
                                   acc_y, acc_z)


if __name__ == "__main__":
    n_type = 3
    sample_input = np.random.randint(1, 10, (n_type, n_type, 4))
    sample_input[:, :, 2] *= -1
    sample_input[:, :, 0] = sample_input[:, :, 1] - 2

    mof = all_force_functions("cluster_distance_input", *sample_input)

    n_type_arr = np.array([1000, 1000, 1000])

    pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, interact_matrix, max_particles = initialise(
        n_type_arr)

    # _plot_dist(pos_x, pos_y, pos_z, max_particles, 3)

    acc_x = np.zeros_like(vel_x)
    acc_y = np.zeros_like(vel_x)
    acc_z = np.zeros_like(vel_x)

    accelerator(mof, pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, (100, 100), 10,
                n_type_arr, max_particles, acc_x, acc_y, acc_z)

    start = time.perf_counter()
    for i in range(10):
        accelerator(mof, pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, (100, 100),
                    10, n_type_arr, max_particles, acc_x, acc_y, acc_z)
    start = time.perf_counter() - start

    print(f"Approx FPS: {10 / start}")
