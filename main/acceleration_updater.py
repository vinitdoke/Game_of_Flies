import time

import numpy as np
from force_profiles import general_force_function
from numba import cuda, njit
from state_parameters import initialise

@cuda.jit(device = True)
def _cuda_general_force_function(profile_type: int, input_vect: float, args: np.ndarray):
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
        if 0 <= input_vect < args[0]:
            return -args[2] / args[0] * input_vect + args[2]
        elif args[0] <= input_vect < (args[0] + args[1]) / 2:
            return 2 * args[3] / (args[1] - args[0]) * (input_vect -
                                                        args[0])
        elif (args[0] + args[1]) / 2 <= input_vect < args[1]:
            return 2 * args[3] / (args[1] - args[0]) * (args[1] -
                                                        input_vect)
        else:
            return 0
    elif profile_type == 1:
        raise NotImplementedError("Position Input not implemented yet")


# Assigns bins to particles and set their offset within it.
# Also finds the number of particles in each bin
@cuda.jit
def bin_particles(
    pos_x: np.ndarray,
    pos_y: np.ndarray,
    #pos_z: np.ndarray,
    num_bin_x: int,
    bin_size_x :float,
    bin_size_y :float,
    num_particles: int,
    particle_bins: np.ndarray,
    particle_bin_counts: np.ndarray,
    bin_offsets: np.ndarray,
    num_bins: int
):
    i = cuda.grid(1)
    if i < num_bins:
        particle_bin_counts[i] = 0
    cuda.syncthreads()
    if i < num_particles:
        particle_bins[i] = (pos_x[i] // bin_size_x) \
              + num_bin_x * (pos_y[i] // bin_size_y)
        bin_offsets[i] = cuda.atomic.add(particle_bin_counts, particle_bins[i], 1)

# Implementation of scan for a cumulative sum
# Reference: https://www.eecs.umich.edu/courses/eecs570/hw/parprefix.pdf
@cuda.jit
def cumsum(values: np.ndarray, result: np.ndarray, d_max: int):
    i = cuda.grid(1)
    n = values.size

    if i < n:
        result[i] = values[i]
    
    cuda.syncthreads()

    for d in range(d_max):
        dp = 1 << d
        dp1 = dp << 1
        j = i * dp1
        if (j + dp1 - 1) < n:
            result[j + dp1 - 1] = result[j + dp - 1] + result[j + dp1 - 1]
        cuda.syncthreads()
    
    
    cuda.syncthreads()
    if i == 0:
        result[n - 1] = 0
    cuda.syncthreads()

    for _ in range(d_max):
        d = d_max - _ - 1
        dp = 1 << d
        dp1 = dp << 1
        j = i * dp1
        if (j + dp1 - 1) < n:
            t = result[j + dp - 1]
            result[j + dp - 1] = result[j + dp1 - 1]
            result[j + dp1 - 1] = t + result[j + dp1 - 1]
        cuda.syncthreads()

# Reorders the particles for usage
@cuda.jit
def set_indices(
    particle_bins: np.ndarray,
    particle_bin_starts: np.ndarray,
    bin_offsets: np.ndarray,
    particle_indices: np.ndarray,
    num_particles: int
):
    i = cuda.grid(1)
    if i < num_particles:
        particle_indices[particle_bin_starts[particle_bins[i]] + bin_offsets[i]] = i


# Called once duing initialization to store which bins are neighbours.
## Stores only half the neighbours for double speed
@njit
def set_bin_neighbours(num_bin_x: int, num_bin_y: int, bin_neighbours: np.ndarray):
    for i in range(num_bin_x):
        ip1 = i + 1
        im1 = i - 1
        if i == 0:
            im1 = num_bin_x - 1
        elif i == num_bin_x - 1:
            ip1 = 0
        for j in range(num_bin_y):
            jp1 = j + 1
            jm1 = j - 1
            if j == 0:
                jm1 = num_bin_y - 1
            elif j == num_bin_y - 1:
                jp1 = 0
            bin_neighbours[i + j * num_bin_x, 0] = i + j * num_bin_x
            bin_neighbours[i + j * num_bin_x, 1] = ip1 + j * num_bin_x
            bin_neighbours[i + j * num_bin_x, 2] = im1 + jp1 * num_bin_x
            bin_neighbours[i + j * num_bin_x, 3] = i + jp1 * num_bin_x
            bin_neighbours[i + j * num_bin_x, 4] = ip1 + jp1 * num_bin_x

# 2D implementation
@cuda.jit
def accelerator(
        pos_x: np.ndarray,
        pos_y: np.ndarray,
        vel_x: np.ndarray,
        vel_y: np.ndarray,
        limitx: float,
        limity: float,
        r_max :float, # max distance at which particles interact
        num_particles: int,
        parameter_matrix: np.ndarray,
        particle_type_index_array: np.ndarray,

        acc_x: np.ndarray,
        acc_y: np.ndarray,

        bin_neighbours: np.ndarray,
        particle_bins: np.ndarray,
        particle_indices: np.ndarray,
        bin_starts: np.ndarray,
        bin_counts: np.ndarray
) -> None:
    i = cuda.grid(1) # The first particle

    if i < num_particles:
        acc_x[i] = 0
        acc_y[i] = 0
    
    cuda.syncthreads()

    if i < num_particles:
        for b in range(5):
            bin2 = bin_neighbours[particle_bins[i], b]
            for p in range(bin_counts[bin2]):
                j = particle_indices[bin_starts[bin2] + p] # The second particle
                if i != j:
                    pos_x_2 = pos_x[j]
                    pos_y_2 = pos_y[j]

                    # Implements periodic BC
                    # assumes r_max < min(limits) / 3
                    if pos_x[i] < r_max and pos_x_2 > limitx - r_max:
                        pos_x_2 -= limitx
                    elif pos_x_2 < r_max and pos_x[i] > limitx - r_max:
                        pos_x_2 += limitx
                    if pos_y[i] < r_max and pos_y_2 > limity - r_max:
                        pos_y_2 -= limity
                    elif pos_y_2 < r_max and pos_y[i] > limity - r_max:
                        pos_y_2 += limity

                    dist = (pos_x[i] - pos_x_2) * (pos_x[i] - pos_x_2) + (pos_y[i] - pos_y_2) * (pos_y[i] - pos_y_2)
                    if 1e-10 < dist < r_max * r_max:
                        dist = dist ** 0.5
                        acc1 = _cuda_general_force_function(
                            parameter_matrix[-1, particle_type_index_array[i], particle_type_index_array[j]],
                            dist, parameter_matrix[:, particle_type_index_array[i], particle_type_index_array[j]]
                        )
                        
                        cuda.atomic.add(acc_x, i, -acc1 * (pos_x[i] - pos_x_2) / dist)
                        cuda.atomic.add(acc_y, i, -acc1 * (pos_y[i] - pos_y_2) / dist)

                        if bin2 != particle_bins[i]:
                            acc2 = _cuda_general_force_function(
                                parameter_matrix[-1, particle_type_index_array[j], particle_type_index_array[i]],
                                dist, parameter_matrix[:, particle_type_index_array[j], particle_type_index_array[i]]
                            )

                            cuda.atomic.add(acc_x, j, acc2 * (pos_x[i] - pos_x_2) / dist)
                            cuda.atomic.add(acc_y, j, acc2 * (pos_y[i] - pos_y_2) / dist)


if __name__ == "__main__":
    init = initialise(np.array([1000]*5), seed = 0)

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

    parameter_matrix[0, :, :] *= 3
    parameter_matrix[0, :, :] += 5

    parameter_matrix[1, :, :] *= 5
    parameter_matrix[1, :, :] += 8

    parameter_matrix[2, :, :] *= 3
    parameter_matrix[2, :, :] -= 10

    parameter_matrix[3, :, :] *= 12
    parameter_matrix[3, :, :] -= 6

    #parameter_matrix[1, :, :] = 6

    # Always attract self
    for i in range(parameter_matrix[0,:,0].size):
        parameter_matrix[3, i, i] = abs(parameter_matrix[3, i, i])
    
    r_max = np.max(parameter_matrix[1, :, :])
    

    threads = 64
    blocks = int(np.ceil(num_particles/threads))

    num_bin_x = int(np.floor(limits[0] / r_max))
    num_bin_y = int(np.floor(limits[1] / r_max))
    bin_size_x = limits[0] / num_bin_x
    bin_size_y = limits[0] / num_bin_y

    bin_neighbours = np.zeros((num_bin_x * num_bin_y, 5), dtype=np.int32)
    set_bin_neighbours(num_bin_x, num_bin_y, bin_neighbours)
    

    i = 1
    while i < num_bin_x * num_bin_y:
        i *= 2

    particle_bin_counts = np.zeros(i, dtype=np.int32)
    numbins = particle_bin_counts.size
    

    particle_bin_starts = np.zeros_like(particle_bin_counts, dtype=np.int32)
    particle_bins = np.zeros_like(pos_x, dtype=np.int32)
    particle_indices = np.zeros_like(pos_x, dtype=np.int32)
    bin_offsets = np.zeros_like(pos_x, dtype=np.int32)
    
    pos_xd = cuda.to_device(pos_x)
    pos_yd = cuda.to_device(pos_y)
    vel_xd = cuda.to_device(vel_x)
    vel_yd = cuda.to_device(vel_y)
    acc_xd = cuda.to_device(acc_x)
    acc_yd = cuda.to_device(acc_y)
    limitsd = cuda.to_device(limits)
    particle_tiad = cuda.to_device(particle_type_index_array)
    parameter_matrixd = cuda.to_device(parameter_matrix)

    bin_offsetsd = cuda.to_device(bin_offsets)
    particle_binsd = cuda.to_device(particle_bins)
    particle_bin_countsd = cuda.to_device(particle_bin_counts)
    particle_bin_startsd = cuda.to_device(particle_bin_starts)
    particle_indicesd = cuda.to_device(particle_indices)
    bin_neighboursd = cuda.to_device(bin_neighbours)

    bin_particles[blocks, threads](pos_xd, pos_yd, num_bin_x, bin_size_x, bin_size_y, num_particles,
                                    particle_binsd, particle_bin_countsd, bin_offsetsd, numbins)

    cumsum[blocks, threads](particle_bin_countsd, particle_bin_startsd, int(np.log2(numbins)))
        
    set_indices[blocks, threads](particle_binsd, particle_bin_startsd, bin_offsetsd, particle_indicesd, num_particles)

    accelerator[blocks, threads](pos_xd, pos_yd, vel_xd, vel_yd, limitsd, r_max, num_particles, parameter_matrixd, particle_tiad,
                                    acc_xd, acc_yd, bin_neighboursd, particle_binsd, bin_offsetsd, particle_indicesd, particle_bin_startsd, particle_bin_countsd)

    reps = 10
    start = time.perf_counter()
    for i in range(reps):
        accelerator[blocks, threads](pos_xd, pos_yd, vel_xd, vel_yd, limitsd, r_max, num_particles, parameter_matrixd, particle_tiad,
                                    acc_xd, acc_yd, bin_neighboursd, particle_binsd, bin_offsetsd,
                                    particle_indicesd, particle_bin_startsd, particle_bin_countsd)
        
        '''acc_xd.copy_to_host(acc_x)
        acc_yd.copy_to_host(acc_y)
        print(acc_x[:10])
        print(acc_y[:10])'''

    start = time.perf_counter() - start

    print(f"Physics time: {start / reps}")

    '''
    particle_bin_countsd.copy_to_host(particle_bin_counts)
    particle_bin_startsd.copy_to_host(particle_bin_starts)
    particle_binsd.copy_to_host(particle_bins)
    bin_offsetsd.copy_to_host(bin_offsets)
    particle_indicesd.copy_to_host(particle_indices)

    print(particle_bins[:num_particles])
    print(particle_bin_counts)
    print(particle_bin_starts)
    print(bin_offsets[:num_particles])
    print(particle_indices[:num_particles])'''

    '''
    accelerator(pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, limits, r_max, num_particles,
                parameter_matrix, particle_type_index_array, acc_x, acc_y, acc_z)

    start = time.perf_counter()
    for i in range(10):
        accelerator(pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, limits, r_max, num_particles,
                    parameter_matrix, particle_type_index_array, acc_x, acc_y, acc_z)
    start = time.perf_counter() - start

    print(f"Physics time: {start / 10}")
    '''
