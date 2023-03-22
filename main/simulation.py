from state_parameters import initialise
from integrator import integrate
import numpy as np


class Simulation:
    def __init__(self, n_type) -> None:
        self.init = initialise(n_type)

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
        self.parameter_matrix[2, :, :] += 2

        self.parameter_matrix[3, :, :] *= 12
        self.parameter_matrix[3, :, :] -= 6

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
    #    print("update_called")
