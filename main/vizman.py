import numpy as np
import vispy
import vispy.scene
from vispy import app
from vispy.color import ColorArray
from vispy.scene import visuals
from vispy.scene.cameras import PanZoomCamera, TurntableCamera

# TODO Rename scatters attribute to scene_objects

class Visualiser:
    def __init__(self):

        self.canvas = vispy.scene.SceneCanvas(keys="interactive", show=True)
        self.view = self.canvas.central_widget.add_view()
        self.scatters = []


        self.simulation = None
        self.axis = None

        self.timer = app.Timer()
        self.timer.connect(self.update)


        # BOOLS
        self.plotting_initialised = False

        # To print FPS, set to True
        self.print_fps = False  # DEFAULT
        self.canvas.measure_fps(window=1, callback=self.blah)

        # Show Canvas
        self.canvas_shown = False


        self.colour_array = None
        self.boundary = None

        self.COLOUR_LIST = [
            "red",
            "blue",
            "green",
            "yellow",
            "orange",
            "purple",
            "pink",
            "brown",
            "white",
        ]

    def blah(self, FPS):
        if self.print_fps:
            print(f"FPS: {FPS:.2f}")

    def create_Camera(self):
        limits = self.simulation.limits
        if limits[2] == 0:
            self.view.camera = PanZoomCamera(
                rect=(0, 0, limits[0], limits[1]),
                aspect=1,
            )
        else:
            self.view.camera = TurntableCamera(
                center=(limits[0] / 2, limits[1] / 2, limits[2] / 2),
                fov=60,
                distance=1.5 * max(limits),
                elevation=30,
            )

    def set_simulation_instance(self, obj):
        self.simulation = obj
        self.create_Camera()
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
            : self.simulation.num_particles
        ]
        colour_array = []
        for index in index_list:
            colour_array.append(self.COLOUR_LIST[index])
        # if index in [0, 2, 8]:
        #     colour_array.append(self.COLOUR_LIST[index])
        # else:
        #     colour_array.append((1,1,1,0))
        self.colour_array = ColorArray(colour_array)

    def update(self, _):
        self.simulation.update()
        #print(self.simulation.output)
        self.scatters[0].set_data(
            self.simulation.output,
            edge_color=None,
            face_color=self.colour_array,
            size=7,
        )

    def get_data(self):
        raise NotImplementedError

    def draw_boundary(self):
        limits = self.simulation.limits

        if limits[2] == 0:
            self.boundary = visuals.Rectangle(
                center=(limits[0] / 2, limits[1] / 2, 0),
                width=limits[0],
                height=limits[1],
                border_color=(1, 1, 1, 0.5),
                color=(1, 1, 1, 0),
                parent=self.view.scene,
            )
        else:
            # INELEGANT :(  (but it works)
            pts = np.array(
                [
                    [0, 0, 0],
                    [0, limits[1], 0],
                    [limits[0], limits[1], 0],
                    [limits[0], 0, 0],
                    [0, 0, 0],
                    [0, 0, limits[2]],
                    [0, limits[1], limits[2]],
                    [0, limits[1], 0],
                    [0, limits[1], limits[2]],
                    [limits[0], limits[1], limits[2]],
                    [limits[0], limits[1], 0],
                    [limits[0], limits[1], limits[2]],
                    [limits[0], 0, limits[2]],
                    [limits[0], 0, 0],
                    [limits[0], 0, limits[2]],
                    [0, 0, limits[2]],
                    [0, 0, 0],
                ]
            )
            self.boundary = visuals.Line(
                pos=pts, color=(1, 1, 1, 0.5), width=1, parent=self.view.scene
            )

    def start(self):
        if not self.canvas_shown:
            self.timer.start(0)
            self.canvas.show()
            self.canvas_shown = True
        else:
            self.timer.start(0)


def dummy_output():
    # 2D output in 3D structure
    out = np.random.uniform(0, 100, (10000, 3))
    out[:, 2] = 0
    return out
