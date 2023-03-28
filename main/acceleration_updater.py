from force_profiles import general_force_function
from state_parameters import initialise
import numpy as np
from numba import njit, prange
import time


<<<<<<< HEAD
@numba.njit
def resolve_particle_group(
        force_function,
        index_1: int, # the indices of the particle types to be resolved
        index_2: int,
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

                distance = (pos_x_1 - pos_x_2) * (pos_x_1 - pos_x_2) + (pos_y_1 - pos_y_2) * (pos_y_1 - pos_y_2)

                if 1e-10 < distance < r_max * r_max: # 1e-10 is r_min**2
                    distance = np.sqrt(distance)
                    acc = force_function(distance)
                    a_x = acc * (pos_x_1 - pos_x_2) / distance
                    a_y = acc * (pos_y_1 - pos_y_2) / distance

                    acc_x[p1] -= a_x
                    acc_y[p1] -= a_y


def accelerator(
        matrix_of_functions: list,
=======

@njit(parallel = True)
def accelerator(
>>>>>>> object_oriented
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
        acc_z: np.ndarray, # to avoid allocating inside function
        r_min_sq: float = 1e-10
) -> None:
    acc_x *= 0
    acc_y *= 0
    acc_z *= 0

<<<<<<< HEAD
    num_particle_types = n_type_array.shape[0]
    for j1 in range(num_particle_types):
        for j2 in range(num_particle_types):
            force_function = matrix_of_functions[j1][j2]
            resolve_particle_group(force_function, j1, j2, pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, (100, 100), 10, n_type_array, max_particles_per_type, acc_x, acc_y, acc_z)


if __name__ == "__main__":
    n_type = 3
    sample_input = np.random.randint(1, 10, (n_type, n_type, 4))
    sample_input[:, :, 2] *= -1
    sample_input[:, :, 0] = sample_input[:, :, 1] - 2

    mof = all_force_functions("cluster_distance_input", *sample_input)

    n_type_arr = np.array([1000, 1000, 1000])

    pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, interact_matrix, max_particles = initialise(n_type_arr)

    #_plot_dist(pos_x, pos_y, pos_z, max_particles, 3)

    acc_x = np.zeros_like(vel_x)
    acc_y = np.zeros_like(vel_x)
    acc_z = np.zeros_like(vel_x)

    accelerator(mof, pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, (100, 100), 10, n_type_arr, max_particles, acc_x, acc_y, acc_z)

    start = time.perf_counter()
    for i in range(10):
        accelerator(mof, pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, (100, 100), 10, n_type_arr, max_particles, acc_x, acc_y, acc_z)
    start = time.perf_counter() - start

    print(f"Approx FPS: {10 / start}")
=======
    for i in prange(num_particles):
        #if particle_type_index_array[i] == 0:
            #continue
        pos_x_1 = pos_x[i]
        pos_y_1 = pos_y[i]
        pos_z_1 = pos_z[i]

        p1 = particle_type_index_array[i]

        for j in prange(num_particles):
            if i != j:
                pos_x_2 = pos_x[j]
                pos_y_2 = pos_y[j]
                pos_z_2 = pos_z[j]

                # Implements periodic BC
                # assumes r_max < min(limits) / 3
                if pos_x_1 < r_max and pos_x_2 > limits[0] - r_max:
                    pos_x_2 -= limits[0]
                elif pos_x_2 < r_max and pos_x_1 > limits[0] - r_max:
                    pos_x_2 += limits[0]
                if pos_y_1 < r_max and pos_y_2 > limits[1] - r_max:
                    pos_y_2 -= limits[1]
                elif pos_y_2 < r_max and pos_y_1 > limits[1] - r_max:
                    pos_y_2 += limits[1]
                if pos_z_1 < r_max and pos_z_2 > limits[2] - r_max:
                    pos_z_2 -= limits[2]
                elif pos_z_2 < r_max and pos_z_1 > limits[2] - r_max:
                    pos_z_2 += limits[2]
                
                distance = (pos_x_1 - pos_x_2) * (pos_x_1 - pos_x_2) + \
                (pos_y_1 - pos_y_2) * (pos_y_1 - pos_y_2) + \
                (pos_z_1 - pos_z_2) * (pos_z_1 - pos_z_2)

                if r_min_sq < distance < r_max * r_max:  # 1e-10 is r_min**2
                    distance = np.sqrt(distance)

                    p2 = particle_type_index_array[j]

                    acc = general_force_function(parameter_matrix[-1, p1, p2],
                                                np.array([distance]),
                                                parameter_matrix[:, p1, p2])
                    a_x = acc * (pos_x_1 - pos_x_2) / distance
                    a_y = acc * (pos_y_1 - pos_y_2) / distance
                    a_z = acc * (pos_z_1 - pos_z_2) / distance

                    acc_x[i] -= a_x
                    acc_y[i] -= a_y
                    acc_z[i] -= a_z

if __name__ == "__main__":
    init = initialise(np.array([1000, 1000, 1000]))

    pos_x = init["pos_x"]
    pos_y = init["pos_y"]
    pos_z = init["pos_z"]
    vel_x = init["vel_x"]
    vel_y = init["vel_y"]
    vel_z = init["vel_z"]
    acc_x = init["acc_x"]
    acc_y = init["acc_y"]
    acc_z = init["acc_z"]
    limits = np.array(init["limits"])
    num_particles = np.sum(init["n_type_array"])
    particle_type_index_array = np.array(init["particle_type_indx_array"], dtype="int32")
    parameter_matrix = init["parameter_matrix"]
    r_max = init["max_rmax"]

    accelerator(pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, limits, r_max, num_particles,
                parameter_matrix, particle_type_index_array, acc_x, acc_y, acc_z)

    start = time.perf_counter()
    for i in range(10):
        accelerator(pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, limits, r_max, num_particles,
                    parameter_matrix, particle_type_index_array, acc_x, acc_y, acc_z)
    start = time.perf_counter() - start

    print(f"Physics time: {start / 10}")
>>>>>>> object_oriented
