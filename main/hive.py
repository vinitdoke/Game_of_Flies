from simulation import Simulation
from vis2d import *


if __name__ == "__main__":

	input_array = np.array([500]*9)  # max 9 types due to color_list
	simulation = Simulation(input_array, seed = 434, limits=(100, 100, 100))
	# seed 4, 10, 100, 50, 69, 35, 434, 954, 1039
	simulation.update()  # dummy call to avoid frame freeze on first update

	visual = Visualiser()
	visual.set_simulation_instance(simulation)
	visual.draw_boundary()
	visual.set_axis()

	visual.start()
