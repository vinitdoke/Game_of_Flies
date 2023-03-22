import matplotlib.pyplot as plt
import numpy as np
import time

from force_profiles import general_force_function
from state_parameters import initialise
from integrator import integrate

# TODO : Benchmark Function
# TODO : Profiling Function

color_list = np.array(['red', 'blue', 'green', 'yellow', 'black', 'orange', 'purple'])

extra_pos_x = np.array([])
extra_pos_y = np.array([])


def _profiling():
    pass


def _benchmark():
    pass


def setup_plotter(particle_type_index_array: np.ndarray, num_particles: int,
                  limits: tuple = (100, 100),
                  dark_mode: bool = True,
                  extra: bool = False):
    if np.max(particle_type_index_array) > len(color_list):
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

    colors = []
    for i in range(num_particles):
        colors.append(color_list[particle_type_index_array[i]])
    temp = np.zeros(num_particles)

    scatters = [ax.scatter(temp, temp, s=2, c=colors)]
    if extra:
        scatters.append(ax.scatter([], [], s=2, c='grey'))
    plt.show()
    return fig, scatters, ax


def update_plot(fig, scatters, ax, pos_x, pos_y, pos_z, num_particles, limits, timing=None):
    if timing[0] is not None or True:
        ax.set_title(f"Phy: {timing[0]:.2f}ms"
                     f"Plot: {timing[1]:.2f}ms")
        # ax.set_title(f"Phy: {timing[0]:.2f}ms {1e3/timing[0] :.1f} FPS "
        #              f"Plot: {timing[1]:.2f}ms {1e3/timing[1]:.1f} FPS")
    scatters[0].set_offsets(np.vstack([pos_x[:num_particles], pos_y[:num_particles]]).T)
    
    if len(scatters) == 2:
        global extra_pos_x, extra_pos_y
        if len(extra_pos_x) == 0:
            extra_pos_x = np.zeros(num_particles * 8)
            extra_pos_y = np.zeros_like(extra_pos_x)
        l = num_particles

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

        scatters[1].set_offsets(np.vstack([extra_pos_x, extra_pos_y]).T)
    fig.canvas.draw()
    fig.canvas.flush_events()


def main():
    # Initialise the state parameters
    n_type_arr = [1000, 100, 100]

    init = initialise(n_type_arr)

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

    

    parameter_matrix[0, :, :] *= 2
    parameter_matrix[0, :, :] += 4

    parameter_matrix[1, :, :] *= 5
    parameter_matrix[1, :, :] += 5

    parameter_matrix[2, :, :] *= 3
    parameter_matrix[2, :, :] += 3

    parameter_matrix[3, :, :] *= 5
    parameter_matrix[3, :, :] += 10

    iterations = 10000

    # Initialise the plotter
    fig, scatters, ax = setup_plotter(particle_type_index_array, num_particles)
    phys_time = 0
    frame_time = 0

    # Run the simulation
    for i in range(iterations):
        start = time.perf_counter()
        dt = np.sqrt(np.max(vel_x * vel_x + vel_y * vel_y))

        if dt > 1e-15:
            dt = 0.2 / dt
        else:
            dt = 0.1
        integrate(pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, limits, r_max, num_particles,
                parameter_matrix, particle_type_index_array, acc_x, acc_y, acc_z, dt*2)
        phys_end = time.perf_counter()
        if i % 1 == 0:
            update_plot(fig, scatters, ax, pos_x, pos_y, None, num_particles,
                        limits, timing = [phys_time, frame_time])
        frame_end = time.perf_counter()
        phys_time = (phys_end - start)
        frame_time = (frame_end - start)

if __name__ == '__main__':
    main()
