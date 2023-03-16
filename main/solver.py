import matplotlib.pyplot as plt
import numpy as np

from force_profiles import all_force_functions
from integrator import integrate
from state_parameters import initialise, _plot_dist


def setup_plotter():
    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_aspect('equal')
    plt.show()
    return fig, ax


def update_plot(fig, ax, pos_x, pos_y, pos_z, max_particles, n_type):
    ax.clear()
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_aspect('equal')
    for i in range(n_type):
        ax.scatter(pos_x[i * max_particles: (i + 1) * max_particles],
                   pos_y[i * max_particles: (i + 1) * max_particles])
    fig.canvas.draw()
    fig.canvas.flush_events()


if __name__ == "__main__":

    # initialise params
    n_type = 3
    sample_input = np.random.randint(5, 10, (n_type, n_type, 4))
    sample_input[:, :, 2] *= -1
    sample_input[:, :, 0] = sample_input[:, :, 1] - 2

    mof = all_force_functions("cluster_distance_input", *sample_input)

    n_type_arr = np.array([100, 100, 100])

    pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, interact_matrix, max_particles = initialise(
        n_type_arr)

    vel_x = np.zeros_like(pos_x)
    vel_y = np.zeros_like(pos_x)
    vel_z = np.zeros_like(pos_x)

    acc_x = np.zeros_like(vel_x)
    acc_y = np.zeros_like(vel_x)
    acc_z = np.zeros_like(vel_x)

    # run simulation
    # plt.figure(figsize=(12, 12), dpi=80)
    fig, ax = setup_plotter()

    for i in range(10000):
        # out = np.vstack([pos_x, pos_y]).T
        # output.append(out)
        dt = np.sqrt(np.max(vel_x * vel_x + vel_y * vel_y))

        if dt > 1e-15:
            dt = 0.4 / dt
        else:
            dt = 0.1

        integrate(mof, pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, (100, 100),
                  10, n_type_arr, max_particles, acc_x, acc_y, acc_z, dt)
        if i % 1 == 0:
            # _plot_dist(pos_x, pos_y, pos_z, max_particles, 3)
            update_plot(fig, ax, pos_x, pos_y, pos_z, max_particles, n_type)
