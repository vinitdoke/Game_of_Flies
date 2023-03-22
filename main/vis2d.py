import vispy
import vispy.scene
from vispy.scene import visuals
from vispy import app
from vispy.scene.cameras import PanZoomCamera
from vispy.color import ColorArray
import numpy as np


def dummy_output():
    # 2D output in 3D structure
    out = np.random.uniform(0, 100, (10000, 3))
    out[:, 2] = 0
    return out


class Visualiser:
    def __init__(self):

        self.canvas = vispy.scene.SceneCanvas(keys='interactive', show=True)
        self.view = self.canvas.central_widget.add_view()
        self.view.camera = PanZoomCamera(rect=(0, 0, 100, 100),
                                         aspect=1,
                                         )
        self.scatters = []

        self.plotting_initialised = False

        self.simulation = None
        self.axis = None
        self.timer = None
        self.colour_array = None

        self.COLOUR_LIST = ['red', 'blue', 'green', 'yellow', 'orange',
                            'purple', 'pink', 'brown', 'black', 'white']

    def set_simulation_instance(self, obj):
        self.simulation = obj
        self.create_scatters(1)
        self.generate_colour_array()
        self.plotting_initialised = True

    def create_scatters(self, n_types):
        for _ in range(n_types):
            self.scatters.append(visuals.Markers())
            self.view.add(self.scatters[-1])

    def set_axis(self):
        self.axis = visuals.XYZAxis(parent=self.view.scene)

    def generate_colour_array(self):
        index_list = self.simulation.particle_type_index_array[
                     :self.simulation.num_particles]
        colour_array = []
        for index in index_list:
            colour_array.append(self.COLOUR_LIST[index])
        self.colour_array = ColorArray(colour_array)

    def update(self, _):
        self.simulation.update()
        # print(self.simulation.output)
        self.scatters[0].set_data(self.simulation.output,
                                  edge_color=None,
                                  face_color=self.colour_array,
                                  size=10)

    def get_data(self):
        raise NotImplementedError

    def start(self):
        self.timer = app.Timer()
        self.timer.connect(self.update)
        self.timer.start(0)
        self.canvas.show()
        app.run()
