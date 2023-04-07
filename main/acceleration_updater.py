import time

import numpy as np
from force_profiles import general_force_function
from numba import cuda, njit
from state_parameters import initialise

# r_min, r_max, f_min, f_max
@cuda.jit(device=True)
def clusters(dist: float, args: np.ndarray):
    if dist < 0 or dist > args[1]:
        return 0
    elif dist < args[0]:
        return -args[2] / args[0] * dist + args[2]
    elif dist < (args[0] + args[1]) / 2:
        return 2 * args[3] / (args[1] - args[0]) * (dist - args[0])
    else:
        return 2 * args[3] / (args[1] - args[0]) * (args[1] - dist)

# r_max, separation, alignment, cohesion
# This implements only separation
@cuda.jit(device=True)
def boids(dist: float, del_x: float, del_y: float, args: np.ndarray):
    if dist < args[0] / 4:
        return (
            -(args[1] * del_x / dist),
            -(args[1] * del_y / dist)
        )
    else:
        return (0, 0)


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

        num_types: int,
        boid_acc_x: np.ndarray,
        boid_acc_y: np.ndarray,
        boid_vel_x: np.ndarray,
        boid_vel_y: np.ndarray,
        boid_counts: np.ndarray,

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
        for j in range(num_types):
            boid_acc_x[i*num_types + j] = 0
            boid_acc_y[i*num_types + j] = 0
            boid_vel_x[i*num_types + j] = 0
            boid_vel_y[i*num_types + j] = 0
            boid_counts[i*num_types + j] = 0
    
    cuda.syncthreads()

    if i < num_particles:
        p1 = particle_type_index_array[i]
        for b in range(5):
            bin2 = bin_neighbours[particle_bins[i], b]
            for p in range(bin_counts[bin2]):
                j = particle_indices[bin_starts[bin2] + p] # The second particle
                if i != j:
                    p2 = particle_type_index_array[j]

                    pos_x_1 = pos_x[i]
                    pos_y_1 = pos_y[i]

                    pos_x_2 = pos_x[j]
                    pos_y_2 = pos_y[j]

                    # Implements periodic BC
                    # assumes r_max < min(limits) / 3

                    if pos_x_2 < r_max and pos_x_1 > limitx - r_max:
                        pos_x_1 -= limitx
                    elif pos_x_1 < r_max and pos_x_2 > limitx - r_max:
                        pos_x_1 += limitx
                    if pos_y_2 < r_max and pos_y_1 > limity - r_max:
                        pos_y_1 -= limity
                    elif pos_y_1 < r_max and pos_y_2 > limity - r_max:
                        pos_y_1 += limity

                    if pos_x[i] < r_max and pos_x_2 > limitx - r_max:
                        pos_x_2 -= limitx
                    elif pos_x_2 < r_max and pos_x[i] > limitx - r_max:
                        pos_x_2 += limitx
                    if pos_y[i] < r_max and pos_y_2 > limity - r_max:
                        pos_y_2 -= limity
                    elif pos_y_2 < r_max and pos_y[i] > limity - r_max:
                        pos_y_2 += limity

                    dist = (pos_x[i] - pos_x_2) * (pos_x[i] - pos_x_2) + (pos_y[i] - pos_y_2) * (pos_y[i] - pos_y_2)
                    if (1e-10 < dist) and (dist < r_max * r_max):
                        dist = dist ** 0.5
                        
                        if parameter_matrix[-1, p1, p2] == 0:
                            acc = clusters(
                                dist, parameter_matrix[:, p1, p2]
                            )
                            cuda.atomic.add(acc_x, i, -acc * (pos_x[i] - pos_x_2) / dist)
                            cuda.atomic.add(acc_y, i, -acc * (pos_y[i] - pos_y_2) / dist)
                        elif parameter_matrix[-1, p1, p2] == 1:
                            acc = boids(
                                dist, pos_x_2 - pos_x[i], pos_y_2 - pos_y[i],
                                parameter_matrix[:, p1, p2]
                            )
                            cuda.atomic.add(acc_x, i, acc[0])
                            cuda.atomic.add(acc_y, i, acc[1])

                            cuda.atomic.add(boid_counts, p2 * num_particles + i, 1)
                            cuda.atomic.add(boid_acc_x, p2 * num_particles + i, pos_x_2)
                            cuda.atomic.add(boid_acc_y, p2 * num_particles + i, pos_y_2)
                            cuda.atomic.add(boid_vel_x, p2 * num_particles + i, vel_x[j])
                            cuda.atomic.add(boid_vel_y, p2 * num_particles + i, vel_y[j])

                        if bin2 != particle_bins[i]:
                            if parameter_matrix[-1, p2, p1] == 0:
                                acc = clusters(
                                    dist, parameter_matrix[:, p2, p1]
                                )
                                cuda.atomic.add(acc_x, j, acc * (pos_x[i] - pos_x_2) / dist)
                                cuda.atomic.add(acc_y, j, acc * (pos_y[i] - pos_y_2) / dist)
                            elif parameter_matrix[-1, p2, p1] == 1:
                                acc = boids(
                                    dist, -pos_x_2 + pos_x[i], -pos_y_2 + pos_y[i],
                                    parameter_matrix[:, p2, p1]
                                )
                                cuda.atomic.add(acc_x, j, acc[0])
                                cuda.atomic.add(acc_y, j, acc[1])
                            
                                cuda.atomic.add(boid_counts, p1 * num_particles + j, 1)
                                cuda.atomic.add(boid_acc_x, p1 * num_particles + j, pos_x_1)
                                cuda.atomic.add(boid_acc_y, p1 * num_particles + j, pos_y_1)
                                cuda.atomic.add(boid_vel_x, p1 * num_particles + j, vel_x[i])
                                cuda.atomic.add(boid_vel_y, p1 * num_particles + j, vel_y[i])
    
    cuda.syncthreads()

    # Cohesive attraction and alignment for boids
    if i < num_particles:
        p1 = particle_type_index_array[i]
        for j in range(num_types):
            c = boid_counts[j * num_particles + i]
            if parameter_matrix[-1, p1, j] == 1 and c > 0:
                cohesion = parameter_matrix[3, p1, j]
                alignment = parameter_matrix[2, p1, j]
                ax = (boid_vel_x[j * num_particles + i] / c)
                ay = (boid_vel_y[j * num_particles + i] / c)
                spd = (vel_x[i] * vel_x[i] + vel_y[i] * vel_y[i]) ** 0.5
                if spd > 10:
                    spd = (ax * vel_x[i] + ay * vel_y[i]) / spd
                else:
                    spd = 0.0

                cuda.atomic.add(acc_x, i,
                    cohesion * ((boid_acc_x[j * num_particles + i] / c) - pos_x[i]) +
                    alignment * (ax - spd * vel_x[i])
                )
                cuda.atomic.add(acc_y, i,
                    cohesion * ((boid_acc_y[j * num_particles + i] / c) - pos_y[i]) +
                    alignment * (ay - spd * vel_y[i])
                )
