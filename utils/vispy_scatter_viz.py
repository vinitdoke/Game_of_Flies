import numpy as np
import vispy
import vispy.scene
from vispy.scene import visuals
from vispy import app
from vispy.scene.cameras import PanZoomCamera

"""
DOCUMENTATION:
https://vispy.org/api/vispy.scene.cameras.panzoom.html
https://vispy.org/gallery/scene/realtime_data/ex01_embedded_vispy.html
"""

canvas = vispy.scene.SceneCanvas(keys='interactive', show=True)
view = canvas.central_widget.add_view()

# fixed camera looking along z-axis
view.camera = PanZoomCamera(rect=(0, 0, 100, 100),
                            aspect=1,
                            )

# view.camera = 'panzoom'

# scatter plot
scatter = visuals.Markers()
scatter2 = visuals.Markers()
view.add(scatter)
view.add(scatter2)

# xyz axis
axis = visuals.XYZAxis(parent=view.scene)


def update(ev):
    global scatter, scatter2
    scatter.set_data(dummy_output(), edge_color=None,
                     face_color=(1, 0, 0, 1), size=10)
    scatter2.set_data(dummy_output(), edge_color=None,
					  face_color=(0, 0, 1, 1), size=10)


timer = app.Timer()
timer.connect(update)
timer.start(0)


def dummy_output():
    # 2D output in 3D structure
    out = np.random.uniform(0, 100, (10000, 3))
    out[:, 2] = 0
    return out


if __name__ == "__main__":
    canvas.show()
    app.run()
