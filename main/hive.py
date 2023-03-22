from simulation import Simulation
from vis2d import *

# TODO set_limits() function in Visualiser class

if __name__ == "__main__":
	input_array = np.array([100] * 9)  # max 9 types due to color_list
	simulation = Simulation(input_array)
	simulation.update()  # dummy call to avoid frame freeze on first update

	visual = Visualiser()
	visual.set_simulation_instance(simulation)
	visual.draw_boundary()

	visual.start()
