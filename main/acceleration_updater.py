import time

import numpy as np
from force_profiles import general_force_function
from numba import cuda, njit
from state_parameters import initialise

# r_min, r_max, f_min, f_max
@cuda.jit(device = True)
def clusters_force(dist: float, args: np.ndarray):
    if dist < 0 or dist > args[1]:
        return 0
    elif dist < args[0]:
        return -args[2] / args[0] * dist + args[2]
    elif dist < (args[0] + args[1]) / 2:
        return 2 * args[3] / (args[1] - args[0]) * (dist -
                                                    args[0])
    else:
        return 2 * args[3] / (args[1] - args[0]) * (args[1] -
                                                    dist)

# alignment, r_max, separation, cohesion
@cuda.jit(device = True)
def boids_force(dist: float, del_vx, del_vy, del_vz, del_x, del_y, del_z, args):
    tmp = args[3]
    d = dist
    if dist < args[1] / 3:
        tmp = args[2]
    return (del_x * tmp / dist + args[0] * del_vx / d,
           del_y * tmp / dist + args[0] * del_vy / d,
           del_z * tmp / dist + args[0] * del_vz / d)


# Assigns bins to particles and set their offset within it.
# Also finds the number of particles in each bin
@cuda.jit
def bin_particles(
    pos_x: np.ndarray,
    pos_y: np.ndarray,
    pos_z: np.ndarray,
    num_bin_x: int,
    num_bin_y: int,
    num_bin_z: int,
    bin_size_x:float,
    bin_size_y:float,
    bin_size_z:float,
    num_particles: int,
    particle_bins: np.ndarray,
    particle_bin_counts: np.ndarray,
    bin_offsets: np.ndarray,
    num_bins: int
):
    if bin_size_z > 1:
        i = cuda.grid(1)
        if i < num_bins:
            particle_bin_counts[i] = 0
        cuda.syncthreads()
        if i < num_particles:
            particle_bins[i] = (pos_x[i] // bin_size_x) \
                  + num_bin_x * (pos_y[i] // bin_size_y) \
                  + num_bin_x * num_bin_y * (pos_z[i] // bin_size_z)
            bin_offsets[i] = cuda.atomic.add(particle_bin_counts, particle_bins[i], 1)
    else:
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

    dp = 1

    for _ in range(d_max):
        dp1 = dp << 1
        j = i * dp1
        if (j + dp1 - 1) < n:
            #result[j + dp1 - 1] = result[j + dp - 1] + result[j + dp1 - 1]
            cuda.atomic.add(result, j + dp1 - 1, result[j + dp - 1])
        dp <<= 1

        cuda.syncthreads()


    cuda.syncthreads()
    if i == 0:
        result[n - 1] = 0
    cuda.syncthreads()

    for _ in range(d_max):
        dp1 = dp
        dp >>= 1

        j = i * dp1
        if (j + dp1 - 1) < n:
            '''t = result[j + dp - 1]
            result[j + dp - 1] = result[j + dp1 - 1]
            result[j + dp1 - 1] = t + result[j + dp1 - 1]'''
            result[j + dp - 1] = cuda.atomic.add(result, j + dp1 - 1, result[j + dp - 1])

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
def set_bin_neighbours(num_bin_x: int, num_bin_y: int, num_bin_z: int, bin_neighbours: np.ndarray):
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
            
            if num_bin_z < 1:
                bin_neighbours[i + j * num_bin_x, 0] = i + j * num_bin_x
                bin_neighbours[i + j * num_bin_x, 1] = ip1 + j * num_bin_x
                bin_neighbours[i + j * num_bin_x, 2] = im1 + jp1 * num_bin_x
                bin_neighbours[i + j * num_bin_x, 3] = i + jp1 * num_bin_x
                bin_neighbours[i + j * num_bin_x, 4] = ip1 + jp1 * num_bin_x
            else:
                for k in range(num_bin_z):
                    kp1 = k + 1
                    km1 = k - 1
                    if k == 0:
                        km1 = num_bin_z - 1
                    elif k == num_bin_z - 1:
                        kp1 = 0
                    bin_neighbours[i + j * num_bin_x + k * num_bin_x * num_bin_y, 0] = i + j * num_bin_x + k * num_bin_x * num_bin_y
                    bin_neighbours[i + j * num_bin_x + k * num_bin_x * num_bin_y, 1] = ip1 + j * num_bin_x + k * num_bin_x * num_bin_y
                    bin_neighbours[i + j * num_bin_x + k * num_bin_x * num_bin_y, 2] = im1 + jp1 * num_bin_x + k * num_bin_x * num_bin_y
                    bin_neighbours[i + j * num_bin_x + k * num_bin_x * num_bin_y, 3] = i + jp1 * num_bin_x + k * num_bin_x * num_bin_y
                    bin_neighbours[i + j * num_bin_x + k * num_bin_x * num_bin_y, 4] = ip1 + jp1 * num_bin_x + k * num_bin_x * num_bin_y

                    bin_neighbours[i + j * num_bin_x + k * num_bin_x * num_bin_y, 5] = ip1 + j * num_bin_x + kp1 * num_bin_x * num_bin_y
                    bin_neighbours[i + j * num_bin_x + k * num_bin_x * num_bin_y, 6] = im1 + j * num_bin_x + kp1 * num_bin_x * num_bin_y
                    bin_neighbours[i + j * num_bin_x + k * num_bin_x * num_bin_y, 7] = i + jp1 * num_bin_x + kp1 * num_bin_x * num_bin_y
                    bin_neighbours[i + j * num_bin_x + k * num_bin_x * num_bin_y, 8] = i + jm1 * num_bin_x + kp1 * num_bin_x * num_bin_y
                    bin_neighbours[i + j * num_bin_x + k * num_bin_x * num_bin_y, 9] = ip1 + jp1 * num_bin_x + kp1 * num_bin_x * num_bin_y
                    bin_neighbours[i + j * num_bin_x + k * num_bin_x * num_bin_y, 10] = ip1 + jm1 * num_bin_x + kp1 * num_bin_x * num_bin_y
                    bin_neighbours[i + j * num_bin_x + k * num_bin_x * num_bin_y, 11] = im1 + jp1 * num_bin_x + kp1 * num_bin_x * num_bin_y
                    bin_neighbours[i + j * num_bin_x + k * num_bin_x * num_bin_y, 12] = im1 + jm1 * num_bin_x + kp1 * num_bin_x * num_bin_y
                    bin_neighbours[i + j * num_bin_x + k * num_bin_x * num_bin_y, 13] = i + j * num_bin_x + kp1 * num_bin_x * num_bin_y

# 2D implementation
@cuda.jit
def accelerator(
        pos_x: np.ndarray,
        pos_y: np.ndarray,
        pos_z: np.ndarray,
        vel_x: np.ndarray,
        vel_y: np.ndarray,
        vel_z: np.ndarray,
        limitx: float,
        limity: float,
        limitz: float,
        r_max :float, # max distance at which particles interact
        num_particles: int,
        parameter_matrix: np.ndarray,
        particle_type_index_array: np.ndarray,

        acc_x: np.ndarray,
        acc_y: np.ndarray,
        acc_z: np.ndarray,

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
        acc_z[i] = 0

    cuda.syncthreads()
    if limitz == 0:
        num_neighbours = 5
    else:
        num_neighbours = 14

    if i < num_particles:
        t1 = particle_type_index_array[i]
        for b in range(num_neighbours):
            bin2 = bin_neighbours[particle_bins[i], b]
            for p in range(bin_counts[bin2]):
                j = particle_indices[bin_starts[bin2] + p] # The second particle
                if i != j:
                    t2 = particle_type_index_array[j]
                    pos_x_2 = pos_x[j]
                    pos_y_2 = pos_y[j]
                    pos_z_2 = pos_z[j]
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
                    if pos_z[i] < r_max and pos_z_2 > limitz - r_max:
                        pos_z_2 -= limitz
                    elif pos_z_2 < r_max and pos_z[i] > limitz - r_max:
                        pos_z_2 += limitz

                    dist = (pos_x[i] - pos_x_2) * (pos_x[i] - pos_x_2) + (pos_y[i] - pos_y_2) * (pos_y[i] - pos_y_2) \
                           + (pos_z[i] - pos_z_2)*(pos_z[i] - pos_z_2)

                    if ((1e-10 < dist) and (dist < r_max * r_max + 10000000)) or True:
                        dist = dist ** 0.5
                        if parameter_matrix[-1, t1, t2] == 0:
                            acc1 = clusters_force(dist, parameter_matrix[:, t1, t2])
                            cuda.atomic.add(acc_x, i, -acc1 * (pos_x[i] - pos_x_2) / dist)
                            cuda.atomic.add(acc_y, i, -acc1 * (pos_y[i] - pos_y_2) / dist)
                            #cuda.atomic.add(acc_z, i, -acc1 * (pos_z[i] - pos_z_2) / dist)
                        elif parameter_matrix[-1, t1, t2] == 1:
                            acc1 = boids_force(dist, vel_x[j] - vel_x[i], vel_y[j] - vel_y[i], vel_z[j] - vel_z[i],
                                               pos_x_2 - pos_x[i], pos_y_2 - pos_y[i], pos_z_2 - pos_z[i],
                                               parameter_matrix[:, t1, t2])
                            cuda.atomic.add(acc_x, i, acc1[0])
                            cuda.atomic.add(acc_y, i, acc1[1])
                            #cuda.atomic.add(acc_z, i, acc1[2])

                        #cuda.atomic.add(acc_x, i, 1)
                        #cuda.atomic.add(acc_y, i, 1)
                        cuda.atomic.add(acc_z, i, 1)

                        if bin2 != particle_bins[i]:
                            if parameter_matrix[-1, t2, t1] == 0:
                                acc2 = clusters_force(dist, parameter_matrix[:, t2, t1])
                                cuda.atomic.add(acc_x, j, acc2 * (pos_x[i] - pos_x_2) / dist)
                                cuda.atomic.add(acc_y, j, acc2 * (pos_y[i] - pos_y_2) / dist)
                                #cuda.atomic.add(acc_z, j, acc2 * (pos_z[i] - pos_z_2) / dist)
                            elif parameter_matrix[-1, t2, t1] == 1:
                                acc2 = boids_force(dist, vel_x[i] - vel_x[j], vel_y[i] - vel_y[j], vel_z[i] - vel_z[j],
                                                   pos_x[i] - pos_x_2, pos_y[i] - pos_y_2, pos_z[i] - pos_z_2,
                                                   parameter_matrix[:, t2, t1])
                                cuda.atomic.add(acc_x, j, acc2[0])
                                cuda.atomic.add(acc_y, j, acc2[1])
                                #cuda.atomic.add(acc_z, j, acc2[2])


                            #cuda.atomic.add(acc_x, j, 1)
                            #cuda.atomic.add(acc_y, j, 1)
                            cuda.atomic.add(acc_z, j, 1)


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
    set_bin_neighbours(num_bin_x, num_bin_y, None, bin_neighbours)


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
