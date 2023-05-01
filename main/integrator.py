import numpy as np
from numba import cuda
from acceleration_updater import accelerator, bin_particles, cumsum, set_indices, set_bin_neighbours
import time
from state_parameters import initialise


fac = 0.9
max_spd = 20

# Implementation of reduction for max function
# WARNING: Manipulates input array. May need to use a copied array
@cuda.jit
def max_reduction(result: np.ndarray, d_max: int, n: int):
    i = cuda.grid(1)

    for d in range(d_max):
        dp = 1 << d
        dp1 = dp << 1
        j = i * dp1
        if (j + dp1 - 1) < n:
            if result[j + dp1 - 1] < result[j + dp - 1]:
                result[j + dp1 - 1] = result[j + dp - 1]
        cuda.syncthreads()

@cuda.jit
def step1(
    vel_x: np.ndarray,
    vel_y: np.ndarray,
    acc_x: np.ndarray,
    acc_y: np.ndarray,
    num_particles: float,
    timestep: float = 0.02
):
    i = cuda.grid(1)
    if i < num_particles:
        vel_x[i] += acc_x[i] * 0.5 * timestep
        vel_y[i] += acc_y[i] * 0.5 * timestep

        spd = vel_x[i] * vel_x[i] + vel_y[i] * vel_y[i]
        if spd > max_spd * max_spd:
            spd = spd ** 0.5
            vel_x[i] *= max_spd / spd
            vel_y[i] *= max_spd / spd

@cuda.jit
def step2(
    pos_x: np.ndarray,
    pos_y: np.ndarray,
    vel_x: np.ndarray,
    vel_y: np.ndarray,
    acc_x: np.ndarray,
    acc_y: np.ndarray,
    particle_tia,
    parameter_matrix,
    num_particles: float,
    timestep: float = 0.02
):
    i = cuda.grid(1)
    if i < num_particles:
        pos_x[i] += (vel_x[i] + 0.5 * acc_x[i] * timestep) * timestep
        pos_y[i] += (vel_y[i] + 0.5 * acc_y[i] * timestep) * timestep

        vel_x[i] += acc_x[i] * 0.5 * timestep
        vel_y[i] += acc_y[i] * 0.5 * timestep

        p = particle_tia[i]
        if parameter_matrix[-1, p, p]==0:
            vel_x[i] *= fac
            vel_y[i] *= fac

@cuda.jit
def boundary_condition(
    pos_x: np.ndarray,
    pos_y: np.ndarray,
    limitx: float,
    limity: float,
    num_particles: float
):
    i = cuda.grid(1)
    if i < num_particles:
        if pos_x[i] < 0:
            pos_x[i] += limitx
        elif pos_x[i] > limitx:
            pos_x[i] -= limitx
        if pos_y[i] < 0:
            pos_y[i] += limity
        elif pos_y[i] > limity:
            pos_y[i] -= limity

@cuda.jit
def set_sq_speed(
    vel_x: np.ndarray,
    vel_y: np.ndarray,
    sq_speed: np.ndarray,
    num_particles: np.int64, u
):
    i = cuda.grid(1)
    if i < num_particles:
        sq_speed[i] = vel_x[i] * vel_x[i] + vel_y[i] * vel_y[i]
    elif i < u:
        sq_speed[i] = 0

@cuda.reduce
def max_reduce(a, b):
    if a > b:
        return a
    else:
        return b

def integrate(
        pos_x: np.ndarray,
        pos_y: np.ndarray,
        vel_x: np.ndarray,
        vel_y: np.ndarray,
        limits: np.ndarray,
        r_max :float, # max distance at which particles interact
        num_particles: np.int64,
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

        sq_speed: np.ndarray,

        bin_neighbours: np.ndarray,
        particle_bins: np.ndarray,
        bin_offsets: np.ndarray,
        particle_indices: np.ndarray,
        bin_starts: np.ndarray,
        bin_counts: np.ndarray,

        blocks: int,
        threads: int,

        timestep: np.float64 = None
) -> None:
    if timestep is None:
        u = 1
        while u < num_particles:
            u = u << 1
        set_sq_speed[blocks, threads](vel_x, vel_y, sq_speed, num_particles, u)
        max_reduction[blocks, threads](sq_speed, int(np.log2(num_particles)), u)
        timestep = sq_speed[u - 1]
        timestep = np.sqrt(timestep)
        if timestep > 1e-6:
            timestep = 0.1 / timestep
        else:
            timestep = 0.1
    
    step1[blocks, threads](vel_x, vel_y, acc_x, acc_y, num_particles, timestep)

    accelerator[blocks, threads](pos_x, pos_y, vel_x, vel_y, limits[0], limits[1], r_max, num_particles, parameter_matrix, particle_type_index_array,
                                acc_x, acc_y, num_types, boid_acc_x, boid_acc_y, boid_vel_x, boid_vel_y, boid_counts,
                                bin_neighbours, particle_bins, particle_indices, bin_starts, bin_counts)
    
    step2[blocks, threads](pos_x, pos_y, vel_x, vel_y, acc_x, acc_y, particle_type_index_array, parameter_matrix, num_particles, timestep)

    boundary_condition[blocks, threads](pos_x, pos_y, limits[0], limits[1], num_particles)


def setup_bins(
        pos_x, pos_y, num_bin_x, bin_size_x, bin_size_y, num_bins, num_particles,
        particle_bins, particle_bin_counts, bin_offsets, particle_bin_starts, particle_indices,
        blocks, threads
):
    bin_particles[blocks, threads](pos_x, pos_y, num_bin_x, bin_size_x, bin_size_y, num_particles,
                                   particle_bins, particle_bin_counts, bin_offsets, num_bins
    )    

    #tmp = np.zeros(512, dtype=np.int32)
    #particle_bin_counts.copy_to_host(tmp)
    #print(f"INSIDE: {tmp}")    
    cumsum[blocks, threads](particle_bin_counts, particle_bin_starts, int(np.log2(num_bins)))
    
    set_indices[blocks, threads](particle_bins, particle_bin_starts, bin_offsets, particle_indices, num_particles)