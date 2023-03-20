import matplotlib.pyplot as plt
import numpy as np

from force_profiles import general_force_function
from state_parameters import initialise
from integrator import integrate

# TODO : Benchmark Function
# TODO : Profiling Function

color_list = ['red', 'blue', 'green', 'yellow', 'black', 'orange', 'purple']

extra_pos_x = np.array([])
extra_pos_y = np.array([])


def _profiling():
    pass


def _benchmark():
    pass


def setup_plotter(n_types: int,
                  limits: tuple = (100, 100),
                  dark_mode: bool = False,
                  extra: bool = False):
    if n_types > len(color_list):
        raise ValueError("Too many types for current color list")

    plt.ion()
    fig = plt.figure(dpi=100)

    ax = fig.add_subplot(111)
    if not extra:
        ax.set_xlim(0, limits[0])
        ax.set_ylim(0, limits[1])
    else:
        ax.set_xlim(-limits[0] * 0.5, 1.5 * limits[0])
        ax.set_ylim(-limits[1] * 0.5, 1.5 * limits[1])

    # turn off ticks
    ax.tick_params(axis='both', which='both', length=0)
    # set aspect ratio
    ax.set_aspect('equal')

    # DARK MODE
    if dark_mode:
        fig.patch.set_facecolor('black')
        ax.set_facecolor('black')
        ax.spines['bottom'].set_color('white')
        ax.spines['top'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.spines['right'].set_color('white')
        # ax.tick_params(axis='x', colors='white')
        # ax.tick_params(axis='y', colors='white')
        ax.yaxis.label.set_color('white')
        ax.xaxis.label.set_color('white')

    scatters = []
    for i in range(n_types):
        scatters.append(ax.scatter([], [], s=2, c=color_list[i]))
    if extra:
        scatters.append(ax.scatter([], [], s=2, c='grey'))
    plt.show()
    return fig, scatters


def update_plot(fig, scatters, pos_x, pos_y, pos_z, max_particles, n_type,
                limits):
    # scatter.clear()
    for i in range(n_type):
        # ax.scatter(pos_x[i * max_particles: (i + 1) * max_particles],
        #            pos_y[i * max_particles: (i + 1) * max_particles])
        scatters[i].set_offsets(
            np.vstack([pos_x[i * max_particles: (i + 1) * max_particles],
                       pos_y[i * max_particles: (i + 1) * max_particles]]).T)
    if len(scatters) > n_type:
        global extra_pos_x, extra_pos_y
        if len(extra_pos_x) == 0:
            extra_pos_x = np.zeros(len(pos_x) * 8)
            extra_pos_y = np.zeros_like(extra_pos_x)
        l = n_type * max_particles

        extra_pos_x[0:l] = pos_x - limits[0]
        extra_pos_x[l:2 * l] = pos_x
        extra_pos_x[2 * l:3 * l] = pos_x + limits[0]
        extra_pos_x[3 * l:4 * l] = pos_x + limits[0]
        extra_pos_x[4 * l:5 * l] = pos_x + limits[0]
        extra_pos_x[5 * l:6 * l] = pos_x
        extra_pos_x[6 * l:7 * l] = pos_x - limits[0]
        extra_pos_x[7 * l:8 * l] = pos_x - limits[0]

        extra_pos_y[0:l] = pos_y - limits[1]
        extra_pos_y[l:2 * l] = pos_y - limits[1]
        extra_pos_y[2 * l:3 * l] = pos_y - limits[1]
        extra_pos_y[3 * l:4 * l] = pos_y
        extra_pos_y[4 * l:5 * l] = pos_y + limits[1]
        extra_pos_y[5 * l:6 * l] = pos_y + limits[1]
        extra_pos_y[6 * l:7 * l] = pos_y + limits[1]
        extra_pos_y[7 * l:8 * l] = pos_y

        scatters[n_type].set_offsets(
            np.vstack([extra_pos_x, extra_pos_y]).T)
    fig.canvas.draw()
    fig.canvas.flush_events()


def main():
    # Initialise the state parameters
    n_type_arr = [100, 100, 100, 100]
    state_dict = initialise(n_type_arr)

    iterations = 10000

    # Initialise the plotter
    fig, scatters = setup_plotter(len(n_type_arr), extra=True)

    # Run the simulation
    for i in range(iterations):
        pass


if __name__ == '__main__':
    main()
