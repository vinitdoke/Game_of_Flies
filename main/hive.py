import numpy as np
from argparse import ArgumentParser
from vispy import app

from simulation import Simulation
from ui_container import MainWindow
from vizman import Visualiser


def parse():
    parser = ArgumentParser()
    parser.add_argument('-b', '--blind', action='store_true', default=False,
                        help='Run simulation without visualisation')
    parser.add_argument('-i', '--ui', action='store_true', default=False,
                        help='Run simulation with UI')
    parser.add_argument('-r', '--record', type=str, default=None,
                        help='Path to directory to store simulation')

    return parser.parse_args()


if __name__ == "__main__":

    args = parse()

    if not args.blind:
        # visual.set_simulation_instance(simulation)
        # visual.draw_boundary()

        if args.ui:
            visual = Visualiser()

            app = app.use_app('pyqt5')
            app.create()

            window = MainWindow(visual)
            window.showMaximized()
            # window.show()
            # window.showFullScreen()
            app.run()
        else:
            
            input_array = np.array([2000]*1)  # max 9 types due to color_list
            simulation = Simulation(input_array, limits=(100, 100, 100), seed=434)
            simulation.update()  # dummy call to avoid frame freeze on first update
            # seed 4, 10, 100, 50, 69, 35, 434, 954, 1039
            visual = Visualiser()
            visual.set_simulation_instance(simulation)
            visual.draw_boundary()
            visual.print_fps = not True
            visual.set_axis()
            visual.start()
            app.run()

    else:
        input_array = np.array([2000]*1)  # max 9 types due to color_list
        simulation = Simulation(input_array, limits=(100, 100, 100), seed=434)
        simulation.update()  # dummy call to avoid frame freeze on first update
        # seed 4, 10, 100, 50, 69, 35, 434, 954, 1039
        simulation.blind_run(1000, args.record)
