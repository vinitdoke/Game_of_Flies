from simulation import Simulation
from vis2d import *

# TODO set_limits() function in Visualiser class

if __name__ == "__main__":
	input_array = np.array([500, 500, 500, 500])
	simulation = Simulation(input_array)

	visual = Visualiser()
	visual.set_axis()
	visual.set_simulation_instance(simulation)

	visual.start()
	# print(simulation.particle_type_index_array[:simulation.num_particles])
