import numpy as np
from numba import cuda
from acceleration_updater import accelerator, bin_particles, cumsum, set_indices, set_bin_neighbours
import time
from state_parameters import initialise

fac = 0.95
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
        vel_z: np.ndarray,
        acc_x: np.ndarray,
        acc_y: np.ndarray,
        acc_z: np.ndarray,
        num_particles: float,
        timestep: float = 0.02
):
    i = cuda.grid(1)
    if i < num_particles:
        vel_x[i] += acc_x[i] * 0.5 * timestep
        vel_y[i] += acc_y[i] * 0.5 * timestep
        vel_z[i] += acc_z[i] * 0.5 * timestep

        spd = vel_x[i] * vel_x[i] + vel_y[i] * vel_y[i] + + vel_z[i] * vel_z[i]
        if spd > max_spd * max_spd:
            spd = spd ** 0.5
            vel_x[i] *= max_spd / spd
            vel_y[i] *= max_spd / spd
            vel_z[i] *= max_spd / spd


@cuda.jit
def step2(
        pos_x: np.ndarray,
        pos_y: np.ndarray,
        pos_z: np.ndarray,
        vel_x: np.ndarray,
        vel_y: np.ndarray,
        vel_z: np.ndarray,
        acc_x: np.ndarray,
        acc_y: np.ndarray,
        acc_z: np.ndarray,
        particle_tia,
        parameter_matrix,
        num_particles: float,
        timestep: float = 0.02
):
    i = cuda.grid(1)
    if i < num_particles:
        pos_x[i] += (vel_x[i] + 0.5 * acc_x[i] * timestep) * timestep
        pos_y[i] += (vel_y[i] + 0.5 * acc_y[i] * timestep) * timestep
        pos_z[i] += (vel_z[i] + 0.5 * acc_z[i] * timestep) * timestep

        vel_x[i] += acc_x[i] * 0.5 * timestep
        vel_y[i] += acc_y[i] * 0.5 * timestep
        vel_z[i] += acc_z[i] * 0.5 * timestep

        p = particle_tia[i]
        if parameter_matrix[-1, p, p] == 0:
            vel_x[i] *= fac
            vel_y[i] *= fac
            vel_z[i] *= fac


@cuda.jit
def boundary_condition(
        pos_x: np.ndarray,
        pos_y: np.ndarray,
        pos_z: np.ndarray,
        limitx: float,
        limity: float,
        limitz: float,
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
        if pos_z[i] < 0:
            pos_z[i] += limitz
        elif pos_z[i] > limitz:
            pos_z[i] -= limitz


@cuda.jit
def set_sq_speed(
        vel_x: np.ndarray,
        vel_y: np.ndarray,
        vel_z: np.ndarray,
        sq_speed: np.ndarray,
        num_particles: np.int64, u
):
    i = cuda.grid(1)
    if i < num_particles:
        sq_speed[i] = vel_x[i] * vel_x[i] + vel_y[i] * vel_y[i] + vel_z[i] * vel_z[i]
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
        pos_z: np.ndarray,
        vel_x: np.ndarray,
        vel_y: np.ndarray,
        vel_z: np.ndarray,
        limits: np.ndarray,
        r_max: float,  # max distance at which particles interact
        num_particles: np.int64,
        parameter_matrix: np.ndarray,
        particle_type_index_array: np.ndarray,

        acc_x: np.ndarray,
        acc_y: np.ndarray,
        acc_z: np.ndarray,

        num_types: int,
        boid_acc_x: np.ndarray,
        boid_acc_y: np.ndarray,
        boid_acc_z: np.ndarray,
        boid_vel_x: np.ndarray,
        boid_vel_y: np.ndarray,
        boid_vel_z: np.ndarray,
        boid_counts: np.ndarray,

        sq_speed: np.ndarray,

        bin_neighbours: np.ndarray,
        particle_bins: np.ndarray,
        particle_indices: np.ndarray,
        bin_starts: np.ndarray,
        bin_counts: np.ndarray,

        blocks: int,
        threads: int,

        timestep: np.float64 = None
) -> None:
    if timestep is None:
        u = 1
        u2 = 0
        while u < num_particles:
            u = u << 1
            u2 += 1
        set_sq_speed[blocks, threads](vel_x, vel_y, vel_z, sq_speed, num_particles, u)
        max_reduction[blocks, threads](sq_speed, u2, u)

        timestep = sq_speed[-1]
        timestep = np.sqrt(timestep)
        if timestep > 1e-6:
            timestep = 0.1 / timestep
        else:
            timestep = 0.1

    step1[blocks, threads](vel_x, vel_y, vel_z, acc_x, acc_y, acc_z, num_particles, timestep)

    accelerator[blocks, threads](pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, limits[0], limits[1], limits[2], r_max,
                                 num_particles, parameter_matrix, particle_type_index_array,
                                 acc_x, acc_y, acc_z, num_types, boid_acc_x, boid_acc_y, boid_acc_z, boid_vel_x,
                                 boid_vel_y, boid_vel_z, boid_counts,
                                 bin_neighbours, particle_bins, particle_indices, bin_starts, bin_counts)

    step2[blocks, threads](pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, acc_x, acc_y, acc_z, particle_type_index_array,
                           parameter_matrix, num_particles, timestep)

    boundary_condition[blocks, threads](pos_x, pos_y, pos_z, limits[0], limits[1], limits[2], num_particles)


def setup_bins(
        pos_x, pos_y, pos_z, num_bin_x, num_bin_y, num_bin_z, bin_size_x, bin_size_y, bin_size_z, num_bins,
        num_particles,
        particle_bins, particle_bin_counts, bin_offsets, particle_bin_starts, particle_indices,
        blocks, threads
):
    bin_particles[blocks, threads](pos_x, pos_y, pos_z, num_bin_x, num_bin_y, num_bin_z, bin_size_x, bin_size_y,
                                   bin_size_z, num_particles,
                                   particle_bins, particle_bin_counts, bin_offsets, num_bins
                                   )

    # tmp = np.zeros(512, dtype=np.int32)
    # particle_bin_counts.copy_to_host(tmp)
    # print(f"INSIDE: {tmp}")
    cumsum[blocks, threads](particle_bin_counts, particle_bin_starts, int(np.log2(num_bins)))

    set_indices[blocks, threads](particle_bins, particle_bin_starts, bin_offsets, particle_indices, num_particles)
