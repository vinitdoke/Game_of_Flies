import numpy as np
import vispy
import vispy.scene
from vispy import app
from vispy.color import ColorArray
from vispy.scene import visuals
from vispy.scene.cameras import PanZoomCamera, TurntableCamera
import os

# TODO Rename scatters attribute to scene_objects

class Visualiser:
    def __init__(self, size = (800, 600), filepath = None):

        self.canvas = vispy.scene.SceneCanvas(keys="interactive", show=False,
                                              size=size)
        self.view = self.canvas.central_widget.add_view()
        self.scatters = []


        self.simulation = None
        self.axis = None

        self.timer = app.Timer()


        if filepath is not None:
            # print('filepath set')
            self.filepath = filepath
            self.image_idx = 0
            self.list_of_frames = os.listdir(os.path.join(self.filepath, "frames"))
            self.timer.connect(self.update_from_file)
        else:
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

    def create_Camera(self, limits=None):
        if self.simulation is not None:
            limits = self.simulation.limits
        
        if limits is None:
            raise ValueError("Limits not set")

        if limits[2] == 0:
            print('PanZoomCamera')
            self.view.camera = PanZoomCamera(
                rect=(0, 0, limits[0], limits[1]),
                aspect=1,
            )
        else:
            print('TurntableCamera')
            self.view.camera = TurntableCamera(
                center=(limits[0] / 2, limits[1] / 2, limits[2] / 2),
                fov=60,
                distance=1.5 * max(limits),
                elevation=30,
            )

    def set_simulation_instance(self, obj):
        self.simulation = obj
        self.init_plotting()
        

    def init_plotting(self, limits=None, index_list=None):
        self.create_Camera(limits)
        self.create_scatters(1)
        self.generate_colour_array(index_list)
        self.plotting_initialised = True

    def create_scatters(self, n_types):
        for _ in range(n_types):
            self.scatters.append(visuals.Markers())
            self.view.add(self.scatters[-1])

    def set_axis(self):
        self.axis = visuals.XYZAxis(parent=self.view.scene)

    def generate_colour_array(self, index_list=None):
        if self.simulation is not None:
            index_list = self.simulation.particle_type_index_array[
                : self.simulation.num_particles
            ]

        if index_list is None:
            raise ValueError("No index list provided")

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
            size=5,
        )

    def blind_update(self, output_data):
        self.scatters[0].set_data(
            output_data,
            edge_color=None,
            face_color=self.colour_array,
            size=5,
        )
    
    def update_from_file(self, _):
        if self.image_idx == len(self.list_of_frames):
            # print('replaying')
            self.image_idx = 0
        data = np.load(os.path.join(self.filepath, "frames", self.list_of_frames[self.image_idx]))
        self.blind_update(data)
        self.image_idx += 1

    def get_render(self, **kwargs):
        return self.canvas.render(**kwargs)

    def draw_boundary(self, limits=None):
        if self.simulation is not None:
            limits = self.simulation.limits
        
        if limits is None:
            raise ValueError("Limits not set")

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
        print('Boundary drawn')

    def start(self):
        if not self.canvas_shown:
            self.canvas.show()
            self.canvas_shown = True
            self.timer.start()
        else:
            self.timer.start()
