from state_parameters import initialise
from integrator import integrate, setup_bins
from acceleration_updater import set_bin_neighbours
import numpy as np
import time
from tqdm import tqdm
from numba import cuda
import os


class Simulation:
    def __init__(self, n_type, seed=None, limits=(100, 100, 0)) -> None:
        # FOR NUMPY ARRAY EXPORTING IN BLIND RUN
        self.record_path = None
        self.export_set = False
        self.seed = seed

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
        self.num_types = np.max(self.particle_type_index_array) + 1
        self.parameter_matrix = self.init["parameter_matrix"]
        self.r_max = self.init["max_rmax"]
        self.sq_speed = np.zeros_like(self.vel_x)

        # r_min, r_max, f_min, f_max
        '''self.parameter_matrix[0, :, :] *= 3
        self.parameter_matrix[0, :, :] += 5

        self.parameter_matrix[1, :, :] *= 5
        self.parameter_matrix[1, :, :] += 8

        self.parameter_matrix[2, :, :] *= 3
        self.parameter_matrix[2, :, :] -= 10

        self.parameter_matrix[3, :, :] *= 12
        self.parameter_matrix[3, :, :] -= 6

        # Always attract self
        for i in range(self.parameter_matrix[0,:,0].size):
            self.parameter_matrix[3, i, i] = abs(self.parameter_matrix[3, i, i])
        
        self.r_max = np.max(self.parameter_matrix[1, :, :])
        '''

        # r_max, separation, alignment, cohesion
        '''self.parameter_matrix[0, :, :] *= 0
        self.parameter_matrix[0, :, :] += 5

        self.parameter_matrix[1, :, :] *= 0
        self.parameter_matrix[1, :, :] += 3

        self.parameter_matrix[2, :, :] *= 0
        self.parameter_matrix[2, :, :] += 3

        self.parameter_matrix[3, :, :] *= 0
        self.parameter_matrix[3, :, :] += 0.05'''

        '''self.parameter_matrix[0, :, :] *= 2
        self.parameter_matrix[0, :, :] += 8

        self.parameter_matrix[1, :, :] *= 4
        self.parameter_matrix[1, :, :] += 6

        self.parameter_matrix[2, :, :] *= 2
        self.parameter_matrix[2, :, :] += 2

        self.parameter_matrix[3, :, :] *= 0.05
        self.parameter_matrix[3, :, :] += 0.1'''

        self.parameter_matrix[-1, :, :] = np.round(np.random.random((self.num_types, self.num_types)))
        self.parameter_matrix[-1, :, :] = 1
        #self.parameter_matrix[-1, :, :] = np.array([[i==j for i in range(self.num_types)] for j in range(self.num_types)])
        self.parameter_matrix[-1, :, :] = 1
        for i in range(self.num_types):
            for j in range(self.num_types):
                if i > j:
                    self.parameter_matrix[-1, j, i] = self.parameter_matrix[-1, i, j]

        boid = self.parameter_matrix[-1, :, :] == 1
        clus = self.parameter_matrix[-1, :, :] == 0
        print(boid)
        
        self.parameter_matrix[0][boid] *= 2
        self.parameter_matrix[0][boid] += 8

        self.parameter_matrix[1][boid] *= 4
        self.parameter_matrix[1][boid] += 2

        self.parameter_matrix[2][boid] *= -2
        self.parameter_matrix[2][boid] -= 2
        f = 7
        self.parameter_matrix[3][boid] *= -0.1
        self.parameter_matrix[3][boid] -= 0.1
        self.parameter_matrix[3][boid] *= f
        for i in range(self.parameter_matrix[0,:,0].size):
            if self.parameter_matrix[-1, i, i] == 1:
                self.parameter_matrix[3, i, i] = abs(self.parameter_matrix[3, i, i]) / f
                self.parameter_matrix[2, i, i] = abs(self.parameter_matrix[2, i, i])

        
        self.parameter_matrix[0][clus] *= 3
        self.parameter_matrix[0][clus] += 5

        self.parameter_matrix[1][clus] *= 5
        self.parameter_matrix[1][clus] += 8

        self.parameter_matrix[2][clus] *= 3
        self.parameter_matrix[2][clus] -= 10

        self.parameter_matrix[3][clus] *= 12
        self.parameter_matrix[3][clus] -= 6

        self.r_max = np.max(self.parameter_matrix[0:1, :, :])

        self.threads = 512 # weird issues with lower no. of threads. Do not reduce
        self.blocks = int(np.ceil(self.num_particles/self.threads))

        self.num_bin_x = int(np.floor(self.limits[0] / self.r_max))
        self.num_bin_y = int(np.floor(self.limits[1] / self.r_max))
        self.bin_size_x = self.limits[0] / self.num_bin_x
        self.bin_size_y = self.limits[0] / self.num_bin_y

        self.bin_neighbours = np.zeros((self.num_bin_x * self.num_bin_y, 5), dtype=np.int32)
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

        self.boid_acc_x = np.zeros(self.num_particles * self.num_types, dtype=np.float32)
        self.boid_acc_y = np.zeros_like(self.boid_acc_x, dtype=np.float32)
        self.boid_vel_x = np.zeros_like(self.boid_acc_x, dtype=np.float32)
        self.boid_vel_y = np.zeros_like(self.boid_acc_x, dtype=np.float32)
        self.boid_counts = np.zeros_like(self.boid_acc_x, dtype=np.int32)
        
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

        self.d_boid_acc_x = cuda.to_device(self.boid_acc_x)
        self.d_boid_acc_y = cuda.to_device(self.boid_acc_y)
        self.d_boid_vel_x = cuda.to_device(self.boid_vel_x)
        self.d_boid_vel_y = cuda.to_device(self.boid_vel_y)
        self.d_boid_counts = cuda.to_device(self.boid_counts)
        #print(self.bin_neighbours)

        self.output = np.zeros((self.num_particles, 3))
    
    def core_step(self):
        setup_bins(self.d_pos_x, self.d_pos_y, self.num_bin_x, self.bin_size_x, self.bin_size_y, self.num_bins, self.num_particles,
                self.d_particle_bins, self.d_particle_bin_counts, self.d_bin_offsets, self.d_particle_bin_starts, self.d_particle_indices,
                self.blocks, self.threads
        )
        integrate(self.d_pos_x, self.d_pos_y, self.d_vel_x, self.d_vel_y,
                self.limits, self.r_max, self.num_particles,
                self.d_parameter_matrix, self.d_particle_tia, self.d_acc_x, self.d_acc_y,
                self.num_types, self.d_boid_acc_x, self.d_boid_acc_y, self.d_boid_vel_x, self.d_boid_vel_y, self.d_boid_counts,
                self.d_sq_speed, self.d_bin_neighbours, self.d_particle_bins, self.d_bin_offsets,
                self.d_particle_indices, self.d_particle_bin_starts, self.d_particle_bin_counts,
                self.blocks, self.threads, timestep = 0.01
        )
        '''self.d_acc_x.copy_to_host(self.acc_x)
        self.d_acc_y.copy_to_host(self.acc_y)
        self.d_particle_bins.copy_to_host(self.particle_bins)
        self.d_particle_bin_counts.copy_to_host(self.particle_bin_counts)
        self.d_particle_bin_starts.copy_to_host(self.particle_bin_starts)
        self.d_particle_indices.copy_to_host(self.particle_indices)'''

        # For debugging, do not remove:
        '''start_test = np.cumsum(self.particle_bin_counts)
        print(np.allclose(start_test[:-1], self.particle_bin_starts[1:]))
        i = 0
        while start_test[i] == self.particle_bin_starts[i+1]:
            i += 1
        print(f"i:{i} len:{len(self.particle_bin_starts)}")
        print(self.particle_bin_counts)
        #print(start_test)
        print(self.particle_bin_starts)'''

        '''print("\n\n\n__________________________________________")
        for tb in range(5):
            print(f"START--  {tb}")
            bin = self.particle_bins[tb]
            print(f"acc x: {self.acc_x[tb]}")
            print(f"acc y: {self.acc_y[tb]}")
            #print(self.particle_bin_counts)
            print(f"bin number: {bin}")
            print(self.particle_indices[self.particle_bin_starts[bin]:self.particle_bin_starts[bin]+self.particle_bin_counts[bin]])
            a = self.bin_neighbours[self.particle_bins[tb], :]
            a2 = np.empty_like(self.bin_neighbours)
            c = 0
            for i in range(self.num_bin_x * self.num_bin_y):
                if self.particle_bins[tb] in self.bin_neighbours[i, :]:
                    a2[c] = self.bin_neighbours[i, :]
                    c += 1
            
            print(f"self neighbours: {a}")
            print(f"other neighbours:  {a2[:c]}")
            
            b = self.particle_bin_counts[a]
            b2 = self.particle_bin_counts[a2[:c, 0]]
            print(f"b:  {b}   sum b:  {np.sum(b)}")
            print(f"b2:  {b2} sum b2:  {np.sum(b2) - self.particle_bin_counts[self.particle_bins[tb]]}")
            print("END--\n\n")'''

    def update(self):
        self.core_step()

        self.d_pos_x.copy_to_host(self.pos_x)
        self.d_pos_y.copy_to_host(self.pos_y)

        self.output[:, 0] = self.pos_x[:self.num_particles]
        self.output[:, 1] = self.pos_y[:self.num_particles]
        self.output[:, 2] = 0


    def record(self):
        if not self.export_set:

            os.mkdir(self.record_path)
            os.mkdir(os.path.join(self.record_path, "frames"))

            np.savez(os.path.join(self.record_path, "state.npz"),
                     type_array=self.particle_type_index_array,
                     limits=self.limits,
                     parameter_matrix=self.parameter_matrix,
                     num_particles=self.num_particles,
                     seed=self.seed)

            self.frame = 0
            self.export_set = True


        # np.save(self.record_path + "/frames/" + f"{self.frame:05}", self.output)
        np.save(os.path.join(self.record_path, "frames", f"{self.frame:05}"), self.output)


    def blind_run(self, n_steps, record=None):
        # total_time = 0
        # i = 1
        # frame_rate_window = 5

        if record is not None:
            self.record_path = record
            self.record()


        start = time.perf_counter()
        #for _ in tqdm(range(n_steps)):
        for _ in range(n_steps):
            # time_start = time.time()

            if record is not None:
                self.update()
                self.record()
                self.frame += 1
            else:
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
