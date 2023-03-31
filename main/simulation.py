from state_parameters import initialise
from integrator import integrate, setup_bins
from acceleration_updater import set_bin_neighbours
import numpy as np
import time
from tqdm import tqdm
from numba import cuda


class Simulation:
    def __init__(self, n_type, seed=None, limits=(100, 100, 0)) -> None:
        self.init = initialise(n_type, seed=seed, limits=limits)  # seed 4, 10, 100, 50, 69, 35

        self.pos_x = self.init["pos_x"]
        self.pos_y = self.init["pos_y"]
        self.pos_z = self.init["pos_z"]
        self.vel_x = self.init["vel_x"]
        self.vel_y = self.init["vel_y"]
        self.vel_z = self.init["vel_z"]
        self.acc_x = self.init["acc_x"]
        self.acc_y = self.init["acc_y"]
        self.acc_z = self.init["acc_z"]
        self.limits = np.array(self.init["limits"])
        self.num_particles = np.sum(self.init["n_type_array"])
        self.particle_type_index_array = np.array(self.init["particle_type_indx_array"], dtype="int32")
        self.parameter_matrix = self.init["parameter_matrix"]
        self.r_max = self.init["max_rmax"]
        self.sq_speed = np.zeros_like(self.vel_x)

        self.parameter_matrix[0, :, :] *= 3
        self.parameter_matrix[0, :, :] += 5

        self.parameter_matrix[1, :, :] *= 5
        self.parameter_matrix[1, :, :] += 8

        self.parameter_matrix[2, :, :] *= 3
        self.parameter_matrix[2, :, :] -= 30

        self.parameter_matrix[3, :, :] *= 12
        self.parameter_matrix[3, :, :] -= 6

        # Always attract self
        for i in range(self.parameter_matrix[0,:,0].size):
            self.parameter_matrix[3, i, i] = abs(self.parameter_matrix[3, i, i]) 

        self.r_max = np.max(self.parameter_matrix[1, :, :])


        self.threads = 32
        self.blocks = int(np.ceil(self.num_particles/self.threads))

        self.num_bin_x = int(np.floor(self.limits[0] / self.r_max))
        self.num_bin_y = int(np.floor(self.limits[1] / self.r_max))
        self.bin_size_x = self.limits[0] / self.num_bin_x
        self.bin_size_y = self.limits[0] / self.num_bin_y

        self.bin_neighbours = np.zeros((self.num_bin_x * self.num_bin_y, 9), dtype=np.int32)
        set_bin_neighbours(self.num_bin_x, self.num_bin_y, self.bin_neighbours)
        

        i = 1
        while i < self.num_bin_x * self.num_bin_y:
            i *= 2

        self.particle_bin_counts = np.zeros(i, dtype=np.int32)
        self.num_bins = self.particle_bin_counts.size
        

        self.particle_bin_starts = np.zeros_like(self.particle_bin_counts, dtype=np.int32)
        self.particle_bins = np.zeros_like(self.pos_x, dtype=np.int32)
        self.particle_indices = np.zeros_like(self.pos_x, dtype=np.int32)
        self.bin_offsets = np.zeros_like(self.pos_x, dtype=np.int32)
        
        self.d_pos_x = cuda.to_device(self.pos_x)
        self.d_pos_y = cuda.to_device(self.pos_y)
        self.d_vel_x = cuda.to_device(self.vel_x)
        self.d_vel_y = cuda.to_device(self.vel_y)
        self.d_acc_x = cuda.to_device(self.acc_x)
        self.d_acc_y = cuda.to_device(self.acc_y)
        self.d_sq_speed = cuda.to_device(self.sq_speed)
        self.d_limits = cuda.to_device(self.limits)
        self.d_particle_tia = cuda.to_device(self.particle_type_index_array)
        self.d_parameter_matrix = cuda.to_device(self.parameter_matrix)

        self.d_bin_offsets = cuda.to_device(self.bin_offsets)
        self.d_particle_bins = cuda.to_device(self.particle_bins)
        self.d_particle_bin_counts = cuda.to_device(self.particle_bin_counts)
        self.d_particle_bin_starts = cuda.to_device(self.particle_bin_starts)
        self.d_particle_indices = cuda.to_device(self.particle_indices)
        self.d_bin_neighbours = cuda.to_device(self.bin_neighbours)

        self.output = np.zeros((self.num_particles, 3))
    
    def core_step(self):
        setup_bins(self.d_pos_x, self.d_pos_y, self.num_bin_x, self.bin_size_x, self.bin_size_y, self.num_bins, self.num_particles,
                self.d_particle_bins, self.d_particle_bin_counts, self.d_bin_offsets, self.d_particle_bin_starts, self.d_particle_indices,
                self.blocks, self.threads
        )
        integrate(self.d_pos_x, self.d_pos_y, self.d_vel_x, self.d_vel_y,
                self.limits, self.r_max, self.num_particles,
                self.d_parameter_matrix, self.d_particle_tia, self.d_acc_x, self.d_acc_y, self.d_sq_speed,
                self.d_bin_neighbours, self.d_particle_bins, self.d_bin_offsets, self.d_particle_indices,
                self.d_particle_bin_starts, self.d_particle_bin_counts,
                self.blocks, self.threads, timestep = 0.01
        )

    def update(self):
        self.core_step()

        self.d_pos_x.copy_to_host(self.pos_x)
        self.d_pos_y.copy_to_host(self.pos_y)

        self.output[:, 0] = self.pos_x[:self.num_particles]
        self.output[:, 1] = self.pos_y[:self.num_particles]
        self.output[:, 2] = 0

    def blind_run(self, n_steps):
        # total_time = 0
        # i = 1
        # frame_rate_window = 5
        start = time.perf_counter()
        #for _ in tqdm(range(n_steps)):
        for _ in range(n_steps):
            # time_start = time.time()
            self.core_step()
            # time_stop = time.time()
            # if i%frame_rate_window == 0:
            #     print(i)
            #     total_time += time_stop-time_start
            #     print("FPS: ", 1/(total_time/10))
            #     total_time = 0
            #     i = 1
            # else:
            #     total_time += time_stop-time_start
            #     i += 1
        print(f"Total time in ms: {1e3 * (time.perf_counter() - start)}")
