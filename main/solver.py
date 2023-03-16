import matplotlib.pyplot as plt
import numpy as np

from force_profiles import all_force_functions
from integrator import integrate
from state_parameters import initialise, _plot_dist

color_list = ['red', 'blue', 'green', 'yellow', 'black', 'orange', 'purple']


def setup_plotter(n_types: int, limits=(100, 100)):
    if n_types > len(color_list):
        raise ValueError("Too many types for current color list")

    plt.ion()
    fig = plt.figure(dpi=100)

    ax = fig.add_subplot(111)
    ax.set_xlim(0, limits[0])
    ax.set_ylim(0, limits[1])
    # turn off ticks
    ax.tick_params(axis='both', which='both', length=0)
    # set aspect ratio
    # ax.set_aspect('equal')

    scatters = []
    for i in range(n_types):
        scatters.append(ax.scatter([], [], s=2, c=color_list[i]))
    plt.show()
    return fig, scatters


def update_plot(fig, scatters, pos_x, pos_y, pos_z, max_particles, n_type):
    # scatter.clear()
    for i in range(n_type):
        # ax.scatter(pos_x[i * max_particles: (i + 1) * max_particles],
        #            pos_y[i * max_particles: (i + 1) * max_particles])
        scatters[i].set_offsets(
            np.vstack([pos_x[i * max_particles: (i + 1) * max_particles],
                       pos_y[i * max_particles: (i + 1) * max_particles]]).T)
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

    pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, interact_matrix, max_particles = \
        initialise(n_type_arr)

    vel_x = np.zeros_like(pos_x)
    vel_y = np.zeros_like(pos_x)
    vel_z = np.zeros_like(pos_x)

    acc_x = np.zeros_like(vel_x)
    acc_y = np.zeros_like(vel_x)
    acc_z = np.zeros_like(vel_x)

    print("Initialised")
    # run simulation
    fig, scatters = setup_plotter(n_type)

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
            update_plot(fig, scatters, pos_x, pos_y, pos_z, max_particles, n_type)
