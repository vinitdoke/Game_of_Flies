import numpy as np
import numba
import matplotlib.pyplot as plt


# TODO: Benchmark individual njitted functions and functions in a python list

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
        # TODO: add a check for the input
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
    mof = all_force_functions("cluster_distance_input", *sample_input)
    print('mof_shape = ', np.shape(mof))
    print(sample_input)

    print(type(mof))
    # for i in range(n_type):
    #     for j in range(n_type):
    #         _plot_force_function(mof[i][j], sample_input[i][j][1], (i, j))
