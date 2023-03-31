import numpy as np
from numba import cuda
from acceleration_updater import accelerator, bin_particles, cumsum, set_indices, set_bin_neighbours
import time
from state_parameters import initialise


fac = 0.9


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

@cuda.jit
def step2(
    pos_x: np.ndarray,
    pos_y: np.ndarray,
    vel_x: np.ndarray,
    vel_y: np.ndarray,
    acc_x: np.ndarray,
    acc_y: np.ndarray,
    num_particles: float,
    timestep: float = 0.02
):
    i = cuda.grid(1)
    if i < num_particles:
        pos_x[i] += (vel_x[i] + 0.5 * acc_x[i] * timestep) * timestep
        pos_y[i] += (vel_y[i] + 0.5 * acc_y[i] * timestep) * timestep

        vel_x[i] += acc_x[i] * 0.5 * timestep
        vel_y[i] += acc_y[i] * 0.5 * timestep

        vel_x[i] *= 0.9
        vel_y[i] *= 0.9

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
        if timestep > 1e-4:
            timestep = 1 / timestep
        else:
            timestep = 0.01
    
    step1[blocks, threads](vel_x, vel_y, acc_x, acc_y, num_particles, timestep)

    accelerator[blocks, threads](pos_x, pos_y, vel_x, vel_y, limits[0], limits[1], r_max, num_particles, parameter_matrix, particle_type_index_array,
                                acc_x, acc_y, bin_neighbours, particle_bins, particle_indices, bin_starts, bin_counts)
    
    step2[blocks, threads](pos_x, pos_y, vel_x, vel_y, acc_x, acc_y, num_particles, timestep)

    boundary_condition[blocks, threads](pos_x, pos_y, limits[0], limits[1], num_particles)


def setup_bins(
        pos_x, pos_y, num_bin_x, bin_size_x, bin_size_y, num_bins, num_particles,
        particle_bins, particle_bin_counts, bin_offsets, particle_bin_starts, particle_indices,
        blocks, threads
):
    bin_particles[blocks, threads](pos_x, pos_y, num_bin_x, bin_size_x, bin_size_y, num_particles,
                                   particle_bins, particle_bin_counts, bin_offsets, num_bins)

    cumsum[blocks, threads](particle_bin_counts, particle_bin_starts, int(np.log2(num_bins)))
    
    set_indices[blocks, threads](particle_bins, particle_bin_starts, bin_offsets, particle_indices, num_particles)


if __name__ == "__main__":
    init = initialise(np.array([10000] * 5), seed = 0)

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
    num_bins = particle_bin_counts.size
    

    particle_bin_starts = np.zeros_like(particle_bin_counts, dtype=np.int32)
    particle_bins = np.zeros_like(pos_x, dtype=np.int32)
    particle_indices = np.zeros_like(pos_x, dtype=np.int64)
    bin_offsets = np.zeros_like(pos_x, dtype=np.int32)
    
    d_pos_x = cuda.to_device(pos_x)
    d_pos_y = cuda.to_device(pos_y)
    d_vel_x = cuda.to_device(vel_x)
    d_vel_y = cuda.to_device(vel_y)
    d_acc_x = cuda.to_device(acc_x)
    d_acc_y = cuda.to_device(acc_y)
    d_limits = cuda.to_device(limits)
    d_particle_tia = cuda.to_device(particle_type_index_array)
    d_parameter_matrix = cuda.to_device(parameter_matrix)

    d_bin_offsets = cuda.to_device(bin_offsets)
    d_particle_bins = cuda.to_device(particle_bins)
    d_particle_bin_counts = cuda.to_device(particle_bin_counts)
    d_particle_bin_starts = cuda.to_device(particle_bin_starts)
    d_particle_indices = cuda.to_device(particle_indices)
    d_bin_neighbours = cuda.to_device(bin_neighbours)

    print(pos_x[:10])

    reps = 0
    start = time.perf_counter()
    
    for i in range(reps + 1):
        if i == 1:
            start = time.perf_counter()

        setup_bins(d_pos_x, d_pos_y, num_bin_x, bin_size_x, bin_size_y, num_bins, num_particles,
                d_particle_bins, d_particle_bin_counts, d_bin_offsets, d_particle_bin_starts, d_particle_indices,
                blocks, threads
        )
        '''d_particle_indices.copy_to_host(particle_indices)
        d_particle_bin_starts.copy_to_host(particle_bin_starts)
        d_bin_offsets.copy_to_host(bin_offsets)
        d_particle_bin_counts.copy_to_host(particle_bin_counts)
        d_particle_bins.copy_to_host(particle_bins)
        print(particle_bin_starts[:])
        print(particle_bin_counts[:])'''

        integrate(d_pos_x, d_pos_y, d_vel_x, d_vel_y,
                d_limits, r_max, num_particles,
                d_parameter_matrix, d_particle_tia, d_acc_x, d_acc_y,
                d_bin_neighbours, d_particle_bins, d_bin_offsets, d_particle_indices, d_particle_bin_starts, d_particle_bin_counts,
                blocks, threads, timestep = 0.1
        )

    start = time.perf_counter() - start

    print(f"Physics time: {1e3 * start / max(reps, 1)}")

    d_pos_x.copy_to_host(pos_x)
    print(pos_x[:10])