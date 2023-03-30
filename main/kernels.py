from numba import cuda, njit, prange
import numpy as np
import time
from math import sqrt
from acceleration_updater import accelerator
from state_parameters import initialise
"""
Tutorials:
https://nyu-cds.github.io/python-numba/05-cuda/

Documentation:
https://numba.pydata.org/numba-doc/latest/cuda/index.html

Kernel List:

Device Functions:

"""

# acceleration_updater cuda kernels
@cuda.jit
def _cuda_accelerator(
        pos_x: np.ndarray,
        pos_y: np.ndarray,
        pos_z: np.ndarray,
        vel_x: np.ndarray,
        vel_y: np.ndarray,
        vel_z: np.ndarray,
        limits: np.ndarray,
        r_max: float,  # max distance at which particles interact
        num_particles: int,
        parameter_matrix: np.ndarray,
        particle_type_index_array: np.ndarray,

        acc_x: np.ndarray,
        acc_y: np.ndarray,
        acc_z: np.ndarray,  # to avoid allocating inside function
        r_min_sq: float = 1e-10
) -> None:
    i = cuda.grid(1)
    if i < num_particles:
        pos_x_1 = pos_x[i]
        pos_y_1 = pos_y[i]
        pos_z_1 = pos_z[i]

        p1 = particle_type_index_array[i]

        for j in range(num_particles):
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

                # distance between particles
                r_sq = (pos_x_1 - pos_x_2) ** 2 + (pos_y_1 - pos_y_2) ** 2 + (
                        pos_z_1 - pos_z_2) ** 2
                
                if r_sq < r_min_sq:
                    r_sq = r_min_sq
                
                if r_sq < r_max*r_max:
                    r = sqrt(r_sq)
                    p2 = particle_type_index_array[j]

                    acc = _cuda_general_force_function(parameter_matrix[-1, p1, p2],
                                                       np.array([r]),
                                                       parameter_matrix[:, p1, p2])

                    acc_x[i] += acc * (pos_x_2 - pos_x_1) / r
                    acc_y[i] += acc * (pos_y_2 - pos_y_1) / r
                    acc_z[i] += acc * (pos_z_2 - pos_z_1) / r






# CUDA General Force Function
@cuda.jit(device = True)
def _cuda_general_force_function(profile_type: int, input_vect: np.ndarray, args: np.ndarray):
    """
    :param profile_type: int
    :   1 : clusters_distance_input
    :   2 : clusters_position_input
    :param input_vect: list
    :   1 : [dist]
    :param args: r_min, r_max, f_min, f_max
    :return: force value
    """
    
    if profile_type == 0:
        if 0 <= input_vect[0] < args[0]:
            return -args[2] / args[0] * input_vect[0] + args[2]
        elif args[0] <= input_vect[0] < (args[0] + args[1]) / 2:
            return 2 * args[3] / (args[1] - args[0]) * (input_vect[0] -
                                                        args[0])
        elif (args[0] + args[1]) / 2 <= input_vect[0] < args[1]:
            return 2 * args[3] / (args[1] - args[0]) * (args[1] -
                                                        input_vect[0])
        else:
            return 0
    elif profile_type == 1:
        raise NotImplementedError("Position Input not implemented yet")

@cuda.jit
def _cuda_distance_matrix(x, y, z, dist):
    """
    :param x: x coordinates of the particles
    :param y: y coordinates of the particles
    :param z: z coordinates of the particles
    :param dist: distance matrix
    :return: distance matrix
    """
    i = cuda.grid(1)
    if i < x.shape[0]:
        for j in range(x.shape[0]):
            dist[i, j] = sqrt((x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2 +
                                 (z[i] - z[j]) ** 2)

@njit(parallel = True)
def _simple_distance_matrix(x,y,z,dist):
    for i in prange(x.shape[0]):
        for j in prange(x.shape[0]):
            dist[i,j] = np.sqrt((x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2 +
                                 (z[i] - z[j]) ** 2)


def _sample_workload(n = 10000, dtype = np.float32):
    x = np.random.randn(n).astype(dtype)
    y = np.random.randn(n).astype(dtype)
    z = np.random.randn(n).astype(dtype)
    dist = np.zeros((n, n), dtype = dtype)
    return x, y, z, dist

if __name__ == "__main__":
    # print(cuda.gpus)
    # print(cuda.detect())

    x, y, z, dist = _sample_workload()
    x_global_mem = cuda.to_device(x)
    y_global_mem = cuda.to_device(y)
    z_global_mem = cuda.to_device(z)
    dist_global_mem = cuda.to_device(dist)

    threadsperblock = 32
    blockspergrid = (x.shape[0] + (threadsperblock - 1)) // threadsperblock
    # CUDA COMPUTE TIME
    start = time.time()
    _cuda_distance_matrix[blockspergrid, threadsperblock](x_global_mem, y_global_mem, z_global_mem, dist_global_mem)
    cuda_dist = dist_global_mem.copy_to_host()
    end = time.time()
    print("CUDA Time: ", end - start)
    # CPU COMPUTE TIME
    start = time.time()
    _simple_distance_matrix(x, y, z, dist)
    end = time.time()
    print("CPU Time: ", end - start)

    print(np.allclose(cuda_dist, dist))

    ## TEST acceleration
    init = initialise(np.array([100,100,100]))

    # pos_x = init["pos_x"]
    # pos_y = init["pos_y"]
    # pos_z = init["pos_z"]
    # vel_x = init["vel_x"]
    # vel_y = init["vel_y"]
    # vel_z = init["vel_z"]
    # acc_x = init["acc_x"]
    # acc_y = init["acc_y"]
    # acc_z = init["acc_z"]
    # limits = np.array(init["limits"])
    # num_particles = np.sum(init["n_type_array"])
    # particle_type_index_array = np.array(init["particle_type_indx_array"], dtype="int32")
    # parameter_matrix = init["parameter_matrix"]
    # r_max = init["max_rmax"]

    # pos_x_global_mem = cuda.to_device(pos_x)
    # pos_y_global_mem = cuda.to_device(pos_y)
    # pos_z_global_mem = cuda.to_device(pos_z)
    # vel_x_global_mem = cuda.to_device(vel_x)
    # vel_y_global_mem = cuda.to_device(vel_y)
    # vel_z_global_mem = cuda.to_device(vel_z)
    # acc_x_global_mem = cuda.to_device(acc_x)
    # acc_y_global_mem = cuda.to_device(acc_y)
    # acc_z_global_mem = cuda.to_device(acc_z)
    # limits_global_mem = cuda.to_device(limits)
    # num_particles_global_mem = cuda.to_device(num_particles)
    # particle_type_index_array_global_mem = cuda.to_device(particle_type_index_array)
    # parameter_matrix_global_mem = cuda.to_device(parameter_matrix)
    # r_max_global_mem = cuda.to_device(r_max)

    # threadsperblock = 32
    # blockspergrid = (pos_x.shape[0] + (threadsperblock - 1)) // threadsperblock

    # CUDA
    # start = time.time()
    # _cuda_accelerator[blockspergrid, threadsperblock](pos_x_global_mem, pos_y_global_mem, pos_z_global_mem, vel_x_global_mem, vel_y_global_mem, vel_z_global_mem,
    #                                                   limits_global_mem, r_max_global_mem, num_particles_global_mem,
    #                                                   parameter_matrix_global_mem, particle_type_index_array_global_mem,
    #                                                   acc_x_global_mem, acc_y_global_mem, acc_z_global_mem, 1e-10)
    # end = time.time()
    # print("CUDA Time: ", end - start)

    # # CPU
    # start = time.time()
    # accelerator(pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, limits, r_max, num_particles,
    #                 parameter_matrix, particle_type_index_array, acc_x, acc_y, acc_z)
    # end = time.time()
    # print("CPU Time: ", end - start)