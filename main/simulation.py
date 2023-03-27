from state_parameters import initialise
from integrator import integrate
import numpy as np


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
        self.particle_type_index_array = np.array(
            self.init["particle_type_indx_array"], dtype="int32")
        self.parameter_matrix = self.init["parameter_matrix"]
        self.r_max = self.init["max_rmax"]

        self.parameter_matrix[0, :, :] *= 3
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

        self.output = np.zeros((self.num_particles, 3))

    def update(self):
        integrate(self.pos_x, self.pos_y, self.pos_z, self.vel_x, self.vel_y,
                  self.vel_z,
                  self.limits, self.r_max, self.num_particles,
                  self.parameter_matrix, self.particle_type_index_array,
                  self.acc_x, self.acc_y, self.acc_z)
        self.output[:, 0] = self.pos_x[:self.num_particles]
        self.output[:, 1] = self.pos_y[:self.num_particles]
        self.output[:, 2] = self.pos_z[:self.num_particles]

    def blind_run(self, n_steps):
        for _ in range(n_steps):
            integrate(self.pos_x, self.pos_y, self.pos_z, self.vel_x, self.vel_y,
                  self.vel_z,
                  self.limits, self.r_max, self.num_particles,
                  self.parameter_matrix, self.particle_type_index_array,
                  self.acc_x, self.acc_y, self.acc_z)
