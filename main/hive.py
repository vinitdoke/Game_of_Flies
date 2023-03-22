from simulation import Simulation
# from vis2d import Visualiser
from vis2d import *

if __name__ == "__main__":
    simulation = Simulation(np.array([500, 500]))
    visual = Visualiser()
    visual.create_scatters(1)
    visual.set_axis()
    # give simulation to visual
    visual.simulation = simulation
    visual.start()