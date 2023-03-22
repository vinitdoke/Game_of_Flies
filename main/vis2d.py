import vispy
import vispy.scene
from vispy.scene import visuals
from vispy import app
from vispy.scene.cameras import PanZoomCamera


class visualiser:
	def __init__(self):
		self.axis = None
		self.timer = None
		self.canvas = vispy.scene.SceneCanvas(keys='interactive', show=True)
		self.view = canvas.central_widget.add_view()
		self.scatters = []
		self.update_set = False

	def create_scatters(self, n_types):
		for _ in range(n_types):
			self.scatters.append(visuals.Markers())
			self.view.add(self.scatters[-1])

	def set_axis(self):
		self.axis = visuals.XYZAxis(parent=self.view.scene)

	def update(self, ev):
		data = self.get_data()
		for scatter in self.scatters:
			pass
		# scatter.set_data(,
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
