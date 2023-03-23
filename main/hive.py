from simulation import Simulation
from vis2d import *


if __name__ == "__main__":
<<<<<<< HEAD
	input_array = np.array([200, 20, 20])  # max 9 types due to color_list
=======
	input_array = np.array([300] * 9)  # max 9 types due to color_list
>>>>>>> 134abd58cb27b740af69a028acbf0f057d8f68e9
	simulation = Simulation(input_array)
	# seed 4, 10, 100, 50, 69, 35, 434, 954, 1039
	simulation.update()  # dummy call to avoid frame freeze on first update

	visual = Visualiser()
	visual.set_simulation_instance(simulation)
	visual.draw_boundary()

	visual.start()
