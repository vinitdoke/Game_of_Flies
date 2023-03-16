import matplotlib.pyplot as plt
import numpy as np

from force_profiles import all_force_functions
from integrator import integrate
from state_parameters import initialise, _plot_dist

color_list = ['red', 'blue', 'green', 'yellow', 'black', 'orange', 'purple']


def setup_plotter(n_types: int, limits=(100, 100), dark_mode=False, extra=False):
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

extra_pos_x = np.array([])
extra_pos_y = np.array([])

def update_plot(fig, scatters, pos_x, pos_y, pos_z, max_particles, n_type, limits):
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
        extra_pos_x[l:2*l] = pos_x
        extra_pos_x[2*l:3*l] = pos_x + limits[0]
        extra_pos_x[3*l:4*l] = pos_x + limits[0]
        extra_pos_x[4*l:5*l] = pos_x + limits[0]
        extra_pos_x[5*l:6*l] = pos_x
        extra_pos_x[6*l:7*l] = pos_x - limits[0]
        extra_pos_x[7*l:8*l] = pos_x - limits[0]

        extra_pos_y[0:l] = pos_y - limits[1]
        extra_pos_y[l:2*l] = pos_y - limits[1]
        extra_pos_y[2*l:3*l] = pos_y - limits[1]
        extra_pos_y[3*l:4*l] = pos_y
        extra_pos_y[4*l:5*l] = pos_y + limits[1]
        extra_pos_y[5*l:6*l] = pos_y + limits[1]
        extra_pos_y[6*l:7*l] = pos_y + limits[1]
        extra_pos_y[7*l:8*l] = pos_y

        scatters[n_type].set_offsets(
            np.vstack([extra_pos_x, extra_pos_y]).T)
    fig.canvas.draw()
    fig.canvas.flush_events()


if __name__ == "__main__":

    # initialise params
    n_type = 3
    sample_input = np.random.randint(6, 10, (n_type, n_type, 4))
    sample_input[:, :, 2] *= -1
    sample_input[:, :, 0] = sample_input[:, :, 1] - 5
    sample_input[:, :, 3] = (sample_input[:, :, 3] - 8) * 8

    '''sample_input[:, :, 0] = 7
    sample_input[:, :, 1] = 15
    sample_input[:, :, 2] = -5
    sample_input[0, 0, 3], sample_input[1, 1, 3] = 16, 9
    sample_input[0, 1, 3] = -8
    sample_input[1, 0, 3] = 10'''


    mof = all_force_functions("cluster_distance_input", *sample_input)

    dummy = np.zeros(15)
    np_dummy = np.array([5, 5, 5])
    integrate(mof, dummy, dummy, dummy, dummy, dummy, dummy, (100, 100), 
            10, np_dummy, 5, dummy, dummy, dummy, 1) # dumb run

    n_type_arr = np.array([300, 300, 300])

    pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, interact_matrix, max_particles = \
        initialise(n_type_arr)

    vel_x = np.zeros_like(pos_x)
    vel_y = np.zeros_like(pos_x)
    vel_z = np.zeros_like(pos_x)

    acc_x = np.zeros_like(vel_x)
    acc_y = np.zeros_like(vel_x)
    acc_z = np.zeros_like(vel_x)

    # run simulation
    fig, scatters = setup_plotter(n_type, dark_mode=True, extra=True)

    print("Initialised")

    for i in range(10000):
        # out = np.vstack([pos_x, pos_y]).T
        # output.append(out)
        dt = np.sqrt(np.max(vel_x * vel_x + vel_y * vel_y))

        if dt > 1e-15:
            dt = 0.5 / dt
        else:
            dt = 0.1

        integrate(mof, pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, (100, 100),
                  np.max(sample_input[:, :, 1]), n_type_arr, max_particles, acc_x, acc_y, acc_z, dt)
        if i % 1 == 0:
            # _plot_dist(pos_x, pos_y, pos_z, max_particles, 3)
            update_plot(fig, scatters, pos_x, pos_y, None, max_particles, n_type, (100, 100))
