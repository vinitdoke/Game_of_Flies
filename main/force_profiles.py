import numpy as np
import numba
import matplotlib.pyplot as plt
from time import perf_counter


# TODO: Benchmark individual njitted functions and functions in a python list

@numba.njit
def _sampleJit(dist):
    if 0 <= dist < 0.2:
        return 1.0 / 0.2 * dist + 1.0
    elif 0.2 <= dist < (0.2 + 2.0) / 2:
        return 2 * 1.0 / (2.0 - 0.2) * (dist - 0.2)
    elif (0.2 + 2.0) / 2 <= dist < 2.0:
        return 2 * 1.0 / (2.0 - 0.2) * (2.0 - dist)
    else:
        return 0


def _benchmark(sample_input):
    """
    :return: None
    """
    n_vals = [10 ** i for i in range(1, 7)]
    timings_mof = []
    timings_jit = []
    i, j = 1, 1
    mof = all_force_functions("cluster_distance_input", *sample_input)

    # dry runs
    mof[i][j](1.0)
    _sampleJit(1.0)

    for n in n_vals:
        dist_input = np.random.uniform(0, sample_input[i][j][1], n)
        start = perf_counter()
        for dist in dist_input:
            mof[i][j](dist)
        end = perf_counter()
        timings_mof.append(end - start)
        dist_input = np.random.uniform(0, 2.0, n)
        start = perf_counter()
        for dist in dist_input:
            _sampleJit(dist)
        end = perf_counter()
        timings_jit.append(end - start)

    plt.loglog(n_vals, timings_mof, '-o', label='mof')
    plt.loglog(n_vals, timings_jit, '-o', label='jit')
    plt.title('Jitted Functions : Individual v/s List')
    plt.xlabel('n')
    plt.ylabel('time')
    plt.legend()
    plt.grid()
    plt.show()
    plt.pause(0.1)
    plt.close()
    # plt.savefig('jit_vs_list.png', dpi=500)


def _plot_force_function(force_function, r_max, ij):
    """
    testing function, not for main use
    :param ij: id of the force function
    :param force_function: function
    :param r_max: float
    :return: None
    """
    x = np.linspace(0, r_max, 1000)
    y = np.array([force_function(i) for i in x])
    plt.title(f'{ij}')
    plt.plot(x, y)
    plt.grid()
    plt.savefig(f'force_profile_{ij}.png')
    plt.clf()


def wrap_clusters_force_distance(*args):
    """
    :param args: r_min, r_max, f_min, f_max
    :return: function
    """

    def clusters_force_distance(dist):
        """
        :param dist: float
        :return: force value
        """
        # TODO: simpler check for region
        # TODO: precalculate the values

        if 0 <= dist < args[0]:
            return -args[2] / args[0] * dist + args[2]
        elif args[0] <= dist < (args[0] + args[1]) / 2:
            return 2 * args[3] / (args[1] - args[0]) * (dist - args[0])
        elif (args[0] + args[1]) / 2 <= dist < args[1]:
            return 2 * args[3] / (args[1] - args[0]) * (args[1] - dist)
        else:
            return 0

    return numba.njit(clusters_force_distance)


def all_force_functions(profile_name: str, *params):
    """
    return a matrix of functions
    """
    matrix_of_functions = []

    if profile_name == "cluster_distance_input":
        n_types = len(params)
        # print('n_types = ', n_types)
        for i in range(n_types):
            matrix_of_functions.append([])
            for j in range(n_types):
                input_params = params[i][j]
                matrix_of_functions[i].append(wrap_clusters_force_distance(
                    *input_params))

    elif profile_name == "cluster_position_input":
        # n_types = len(params)
        pass

    elif profile_name == "null_profile":
        pass

    return matrix_of_functions


if __name__ == "__main__":
    n_type = 3
    sample_input = np.random.randint(1, 10, (n_type, n_type, 4))
    sample_input[:, :, 2] *= -1
    sample_input[:, :, 0] = sample_input[:, :, 1] - 2
    # sample_input = [[[1, 2, 3, 4], [5, 6, 7, 8]],
    #                 [[9, 10, 11, 12], [13, 14, 15, 16]]]
    # mof = all_force_functions("cluster_distance_input", *sample_input)
    # print('mof_shape = ', np.shape(mof))
    # print(sample_input)

    # print(type(mof))
    _benchmark(sample_input)
    # for i in range(n_type):
    #     for j in range(n_type):
    #         _plot_force_function(mof[i][j], sample_input[i][j][1], (i, j))
