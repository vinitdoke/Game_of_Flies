from simulation import Simulation
from vizman import *
from vispy import app

from argparse import ArgumentParser


def parse():
	parser = ArgumentParser()
	parser.add_argument('-b', '--blind', action='store_true', default=False)

	return parser.parse_args()

if __name__ == "__main__":

	args = parse()
	input_array = np.array([100]*9)  # max 9 types due to color_list
	simulation = Simulation(input_array, seed = 1234, limits=(100, 100, 100))
	simulation.update()  # dummy call to avoid frame freeze on first update
	# seed 4, 10, 100, 50, 69, 35, 434, 954, 1039

	if not args.blind:
		visual = Visualiser()
		visual.set_simulation_instance(simulation)
		visual.draw_boundary()
		visual.set_axis()
		visual.print_fps = True

		visual.start()
		app.run()
	
	else:
		simulation.blind_run(1000)
