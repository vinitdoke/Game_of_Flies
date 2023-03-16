# def initialise(
#         input_list: list,
#
# ):
#     """
#     Outputs:
#     1. Positions (pos_x, pos_y, pos_z)
#     2. Velocities (vel_x, vel_y, vel_z)
#     3. Type-Indices
#     4. Parameter_matrix (n_type*n_type*3 (r_max, r_min, f_max) array)
#        r_min < r_max
#        r_min != 0
#        0 < f_max <= 1
#        -1 <= f_min < 0
#     """
#     pass

# TODO max_particles -- max_particles_per_type
# TODO initialise -- random_init, params -- random generated
# TODO fix inter shape and name
# TODO change all names to full
# TODO return dictionary of all data
# TODO range defualt
# TODO all arrays as ndarray


import numpy as np
from numpy import random
import matplotlib.pyplot as plt


def _plot_dist(pos_x, pos_y, pos_z,
               max_particles, len_p_tynum):
    plt.clf()
    for i in range(len_p_tynum):
        plt.scatter(pos_x[i * max_particles: (i + 1) * max_particles],
                    pos_y[i * max_particles: (i + 1) * max_particles],
                    label=f"Particle {i}")
    plt.xlim(-1, 101)
    plt.ylim(-1, 101)
    plt.title('Particle Distribution')
    # plt.legend()
    plt.draw()
    plt.pause(0.1)


def initialise(n_type, *params):  # p_tnum -- n_type
    max_particles = 1000  # max particle of a single kind
    n_params = len(params)
    len_p_tynum = len(n_type)
    dist_params = int(n_params / (len_p_tynum * len_p_tynum))

    pos_x = np.empty(len_p_tynum * max_particles) * np.NAN
    pos_y = np.empty(len_p_tynum * max_particles) * np.NAN
    pos_z = np.empty(len_p_tynum * max_particles) * np.NAN

    vel_x = np.empty(len_p_tynum * max_particles) * np.NAN
    vel_y = np.empty(len_p_tynum * max_particles) * np.NAN
    vel_z = np.empty(len_p_tynum * max_particles) * np.NAN

    interact_matrix = np.zeros((dist_params, len_p_tynum, len_p_tynum))

    """Random Initialization of positions and velocities
        num of single kind of particles <= max_particles
        left spaces filled with NAN
    """
    for i in range(len_p_tynum):
        n_specp = int(n_type[i])
        x = 100.0 * random.rand(n_specp)
        y = 100.0 * random.rand(n_specp)
        z = 100.0 * random.rand(n_specp)
        pos_x[i * max_particles: i * max_particles + n_specp] = x[:]
        pos_y[i * max_particles: i * max_particles + n_specp] = y[:]
        pos_z[i * max_particles: i * max_particles + n_specp] = z[:]
        v_x = 2 * random.rand(n_specp) - 1
        v_y = 2 * random.rand(n_specp) - 1
        v_z = 2 * random.rand(n_specp) - 1
        vel_x[i * max_particles: i * max_particles + n_specp] = v_x
        vel_y[i * max_particles: i * max_particles + n_specp] = v_y
        vel_z[i * max_particles: i * max_particles + n_specp] = v_z

    for k in range(dist_params):
        for i in range(len_p_tynum):
            for j in range(len_p_tynum):
                interact_matrix[k, i, j] = params[
                    len_p_tynum * len_p_tynum * k + len_p_tynum * i + j
                    ]

    # _plot_dist(pos_x, pos_y, pos_z, max_particles, len_p_tynum)

    return pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, interact_matrix, max_particles


if __name__ == '__main__':
    p_x, p_y, p_z, v_x, v_y, v_z, inter = initialise([30, 40, 20],
                                                     1, 2, 3,
                                                     4, 5, 6,
                                                     7, 8, 9,
                                                     10, 11, 12,
                                                     13, 14, 15,
                                                     16, 17, 18)
    print(inter.shape)
