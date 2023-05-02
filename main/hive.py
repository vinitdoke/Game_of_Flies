import numpy as np
from argparse import ArgumentParser
from vispy import app
import sys
from simulation import Simulation
from ui_container import MainWindow
from vizman import Visualiser



def parse():
    parser = ArgumentParser()
    parser.add_argument('-b', '--blind', action='store_true', default=False,
                        help='Run simulation without visualisation')
    parser.add_argument('-i', '--ui', action='store_true', default=False,
                        help='Run simulation with UI')
    parser.add_argument('-be', '--benchmark', action='store_true', default=False,
                        help='For Benchmark')
    parser.add_argument('-nt', '--ntype', type=int, nargs='+', default=np.array([100, 100]), help='num of particles')
    parser.add_argument('-o', default='False', help='output file')
    t = sys.argv
    for i,s in enumerate(t):
        t[i] = s.replace("=", " ")
        t[i] = t[i].split()
    tt = []
    for i1 in t:
        for i2 in i1:
            tt.append(i2)
    return parser.parse_args(tt[1:])


if __name__ == "__main__":

    args = parse()
    input_array = np.array([1000] * 9)  # max 9 types due to color_list
    simulation = Simulation(input_array, seed=1234, limits=(100, 100, 100))
    simulation.update()  # dummy call to avoid frame freeze on first update
    # seed 4, 10, 100, 50, 69, 35, 434, 954, 1039

    if not args.blind and (not args.benchmark):

        visual = Visualiser()
        visual.set_simulation_instance(simulation)
        visual.draw_boundary()

        if args.ui:
            app = app.use_app('pyqt5')
            app.create()

            window = MainWindow(visual)
            window.showMaximized()
            # window.show()
            # window.showFullScreen()
            app.run()
        else:
            visual.print_fps = True
            visual.set_axis()
            visual.start()
            app.run()
    elif args.benchmark:
        input_array = args.ntype  # max 9 types due to color_list
        simulation = Simulation(input_array, limits=(100, 100, 0), seed=434)
        simulation.update()  # dummy call to avoid frame freeze on first update
        # seed 4, 10, 100, 50, 69, 35, 434, 954, 1039
        simulation.bench_run(500)
    else:
        simulation.blind_run(1000)
