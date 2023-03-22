import vispy
import vispy.scene
from vispy.scene import visuals
from vispy import app
from vispy.scene.cameras import PanZoomCamera
import numpy as np

def dummy_output():
    # 2D output in 3D structure
    out = np.random.uniform(0, 100, (10000, 3))
    out[:, 2] = 0
    return out

class Visualiser:
	def __init__(self):
		self.axis = None
		self.timer = None
		self.canvas = vispy.scene.SceneCanvas(keys='interactive', show=True)
		self.view = self.canvas.central_widget.add_view()
		self.view.camera = PanZoomCamera(rect=(0, 0, 100, 100),
                            aspect=1,
                            )
		self.scatters = []
		self.update_set = False
		self.simulation = None

    # def set_simulation_instance(self, obj):
    #     self.simulation = obj
	
	def create_scatters(self, n_types):
		for _ in range(n_types):
			self.scatters.append(visuals.Markers())
			self.view.add(self.scatters[-1])

	def set_axis(self):
		self.axis = visuals.XYZAxis(parent=self.view.scene)

	def update(self, ev):
		self.simulation.update()
		# print(self.simulation.output)
		self.scatters[0].set_data(self.simulation.output,
                     edge_color=None,
                     face_color=(1, 0, 0, 1), size=10)
		# Step1 
        # self.simulation.update()
        # # Step2
		# # data = self.simulation
		# for scatter in self.scatters:
		# 	pass
		# scatter.set_data(np.3darray,
		# 				 edge_color=None,
		# 				 face_color=(1, 0, 0, 1), size=10)

	def get_data(self):
		raise NotImplementedError

	def start(self):
		self.timer = app.Timer()
		self.timer.connect(self.update)
		self.timer.start(0)
		self.canvas.show()
		app.run()