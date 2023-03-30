import time

import numpy as np
from force_profiles import general_force_function
from numba import cuda, njit, prange
from state_parameters import initialise

# 2D implementation
@cuda.jit
def bin_particles(
    pos_x: np.ndarray,
    pos_y: np.ndarray,
    #pos_z: np.ndarray,
    num_bin_x: int,
    bin_size :float,
    num_particles: int,
    particle_bins: np.ndarray,
    particle_bin_counts: np.ndarray,
    bin_offsets: np.ndarray
):
    i = cuda.grid(1)
    if i < num_particles:
        particle_bins[i] = (pos_x[i] // bin_size) \
              + num_bin_x * (pos_y[i] // bin_size)
        bin_offsets[i] = cuda.atomic.add(particle_bin_counts, particle_bins[i], 1)

@cuda.jit
def cumsum(values: np.ndarray, result: np.ndarray, d_max: int, tmp: np.ndarray):
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
    result[n - 1] = 0

    for _ in range(d_max):
        d = d_max - _ - 1
        dp = 1 << d
        dp1 = dp << 1
        j = i * dp1
        if (j + dp1 - 1) < n:
            tmp[i] = result[j + dp - 1]
            result[j + dp - 1] = result[j + dp1 - 1]
            result[j + dp1 - 1] = tmp[i] + result[j + dp1 - 1]
        cuda.syncthreads()

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


@njit(parallel = True)
def accelerator(
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
    init = initialise(np.array([10, 10, 10]), seed = 0)

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



    particle_bins = np.zeros_like(pos_x, dtype=np.int32)
    particle_indices = np.zeros_like(pos_x, dtype=np.int32)
    bin_offsets = np.zeros_like(pos_x, dtype=np.int32)

    threads = 128
    blocks = int(np.ceil(num_particles/threads))

    num_bin_x = int(np.ceil(0.5 * limits[0] / r_max))
    num_bin_y = int(np.ceil(0.5 * limits[1] / r_max))

    i = 1
    while i < num_bin_x * num_bin_y:
        i *= 2

    particle_bin_counts = np.zeros(i, dtype=np.int32)
    numbins = particle_bin_counts.size
    particle_bin_starts = np.zeros_like(particle_bin_counts, dtype=np.int32)



    pos_xd = cuda.to_device(pos_x)
    pos_yd = cuda.to_device(pos_y)
    bin_offsetsd = cuda.to_device(bin_offsets)
    particle_binsd = cuda.to_device(particle_bins)
    particle_bin_countsd = cuda.to_device(particle_bin_counts)
    particle_bin_startsd = cuda.to_device(particle_bin_starts)
    particle_indicesd = cuda.to_device(particle_indices)
    tmp = cuda.to_device(np.zeros(numbins))



    bin_particles[blocks, threads](pos_xd, pos_yd, num_bin_x, 2 * r_max, num_particles,
                                   particle_binsd, particle_bin_countsd, bin_offsetsd)

    cumsum[blocks, threads](particle_bin_countsd, particle_bin_startsd, int(np.log2(numbins)), tmp)
    
    set_indices[blocks, threads](particle_binsd, particle_bin_startsd, bin_offsetsd, particle_indicesd, num_particles)


    particle_bin_countsd.copy_to_host(particle_bin_counts)
    particle_bin_startsd.copy_to_host(particle_bin_starts)
    particle_binsd.copy_to_host(particle_bins)
    bin_offsetsd.copy_to_host(bin_offsets)
    particle_indicesd.copy_to_host(particle_indices)

    print(particle_bins[:num_particles])
    print(particle_bin_counts)
    print(particle_bin_starts)
    print(bin_offsets[:num_particles])
    print(particle_indices[:num_particles])

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
