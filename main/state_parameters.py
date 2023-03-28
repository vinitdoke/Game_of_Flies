import numpy as np
from numpy import random


def rand_param_matrix(
        n_type: np.ndarray,
        force_type: str = 'Clusters',
        max_param: int = 4,
        seed: int = None
):
    n = len(n_type)
    n_sq = n * n
    param_matrix = np.zeros((max_param + 1, n, n))

    # last matrix distinguishes the type of interaction
    if force_type == 'Clusters':
        param_matrix[-1][:][:] = 0

    # r_min, r_max, f_min, f_max
    if seed is not None:
        random.seed(seed)
    raw_data = random.rand(n * n * max_param)
    raw_data[:2 * n_sq].sort()
    random.shuffle(raw_data[: n_sq])
    random.shuffle(raw_data[n_sq: 2 * n_sq])

    for i in range(len(param_matrix) - 1):
        param_matrix[i][:][:] = raw_data[i * n_sq: i * n_sq + n_sq].reshape(
            (n, n))

    param_matrix[2][:][:] = -param_matrix[2][:][:]
    r_rmax = np.max(param_matrix[3][:][:])

    return param_matrix, r_rmax


def initialise(
        n_type: np.ndarray,
        seed: int = None,
        limits: tuple = (100, 100, 0)
):
    max_particle = 10  # max particle of a single kind
    buffer = 0  # extra space for adding particles
    total_len = len(n_type) * max_particle + buffer
    total_given_part = sum(n_type)

    total_len = total_given_part * 2

    dim = 3

    # if max(n_type) > max_particle:
    # raise Exception(f"Max allowed num of each particle is {max_particle}")

    """
    Random initialization of positions,
    acceleration and velocities set to 0.
    """
    pos = np.ones((dim, total_len)) * np.nan
    vel = np.ones((dim, total_len)) * np.nan
    acc = np.ones((dim, total_len)) * np.nan

    if seed is not None:
        random.seed(seed)
    rand_data = random.rand(dim, total_len)
    # zero_arr = np.zeros(total_len)

    for i in range(dim):
        pos[i, :total_given_part] = limits[i] * rand_data[i][:total_given_part]
        vel[i, :total_given_part] = np.zeros(total_given_part)
        acc[i, :total_given_part] = np.zeros(total_given_part)

    """
    Particle index array
    """
    part_type_indx_arr = np.ones(total_len) * np.nan
    s = len(n_type) - 1
    for i in range(len(n_type)):
        trick_sum = sum(n_type[:len(n_type) - i])
        part_type_indx_arr[:trick_sum] = s
        s -= 1

    param_matrix, max_rmax = rand_param_matrix(n_type, seed=seed)
    # Dictionary to store all the data
    state_variable_dict = {
        "pos_x": pos[0],
        "pos_y": pos[1],
        "pos_z": pos[2],
        "vel_x": vel[0],
        "vel_y": vel[1],
        "vel_z": vel[2],
        "acc_x": acc[0],
        "acc_y": acc[1],
        "acc_z": acc[2],
        "limits": limits,
        "n_type_array": n_type,
        "particle_type_indx_array": part_type_indx_arr,
        "max_particle_per_type": max_particle,
        "parameter_matrix": param_matrix,
        "max_rmax": max_rmax
    }

    return state_variable_dict


if __name__ == '__main__':
    print(initialise([10, 6, 7]))
