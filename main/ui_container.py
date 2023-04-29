from PyQt5 import QtWidgets, QtGui, QtCore
from vizman import Visualiser
from simulation import Simulation
from vispy.app import use_app


# CANVAS_SIZE = (800, 600)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, visual):
        super().__init__()
        central_widget = QtWidgets.QWidget()
        main_layout = QtWidgets.QHBoxLayout()

        self._canvas = visual
        self._controls = ControlsWidget(_canvas=self._canvas)

        main_layout.addWidget(self._controls, 1)
        main_layout.addWidget(self._canvas.canvas.native, 10)

        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        # COSMETIC
        self.setWindowTitle("Game of Flies")
        self.setWindowIcon(QtGui.QIcon("icon.png"))


class ControlsWidget(QtWidgets.QWidget):
    def __init__(self, parent=None, _canvas=None):
        super().__init__(parent)

        self._canvas = _canvas
        layout = QtWidgets.QVBoxLayout()

        # Start/Stop Button
        self._start_stop_button = QtWidgets.QPushButton("Start")
        self._start_stop_button.clicked.connect(self._start_stop)

        # Profile Selector:
        self._select_profile = QtWidgets.QComboBox()
        self._select_profile.addItems(["Clusters + Boids"])

        # Seed input
        self._seed_label = QtWidgets.QLabel("Seed: ")
        self._seed_label.setAlignment(QtCore.Qt.AlignLeft)
        # self._seed_label.setStyleSheet('font-size: 20px; font-weight: bold')
        self._seed_input = QtWidgets.QLineEdit()
        self._seed_input.setPlaceholderText("Seed")
        self._seed_input.setValidator(QtGui.QIntValidator())

        # Number of flies input
        self._num_flies_label = QtWidgets.QLabel("Number of Flies:")
        self._num_flies_label.setAlignment(QtCore.Qt.AlignLeft)
        # self._num_flies_label.setStyleSheet('font-size: 20px; font-weight: bold')
        self._num_flies_input = QtWidgets.QLineEdit()
        self._num_flies_input.setPlaceholderText("num_particles")
        self._num_flies_input.setValidator(QtGui.QIntValidator())
        # self._num_flies_input.textChanged.connect(self._canvas.update_num_flies)

        # Clusters to Boid Ratio Slider
        self._cluster_boid_ratio_label = QtWidgets.QLabel("Cluster to Boid Ratio:")
        self._cluster_boid_ratio_label.setAlignment(QtCore.Qt.AlignLeft)
        # self._cluster_boid_ratio_label.setStyleSheet('font-size: 20px; font-weight: bold')
        self._cluster_boid_ratio_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self._cluster_boid_ratio_slider.setMinimum(0)
        self._cluster_boid_ratio_slider.setMaximum(20)
        self._cluster_boid_ratio_slider.setValue(0)
        self._cluster_boid_ratio_slider.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self._cluster_boid_ratio_slider.setTickInterval(5)
        self._cluster_boid_ratio_slider.setSingleStep(5)
        # self._cluster_boid_ratio_slider.valueChanged.connect(self._canvas.update_cluster_boid_ratio)

        # Cluster Types:
        self._cluster_types_label = QtWidgets.QLabel("Cluster Types:")
        self._cluster_types_label.setAlignment(QtCore.Qt.AlignLeft)
        # self._cluster_types_label.setStyleSheet('font-size: 20px; font-weight: bold')
        self._cluster_types_input = QtWidgets.QLineEdit()
        self._cluster_types_input.setPlaceholderText("cluster_types")
        self._cluster_types_input.setValidator(QtGui.QIntValidator())
        # self._cluster_types_input.textChanged.connect(self._canvas.update_cluster_types)

        # Boid Types:
        self._boid_types_label = QtWidgets.QLabel("Boid Types:")
        self._boid_types_label.setAlignment(QtCore.Qt.AlignLeft)
        # self._boid_types_label.setStyleSheet('font-size: 20px; font-weight: bold')
        self._boid_types_input = QtWidgets.QLineEdit()
        self._boid_types_input.setPlaceholderText("boid_types")
        self._boid_types_input.setValidator(QtGui.QIntValidator())
        # self._boid_types_input.textChanged.connect(self._canvas.update_boid_types)

        # Interaction Matrix:
        self._interaction_matrix_label = QtWidgets.QLabel("Interaction Matrix:")
        self._interaction_matrix_label.setAlignment(QtCore.Qt.AlignLeft)
        # self._interaction_matrix_label.setStyleSheet('font-size: 20px; font-weight: bold')






        # FPS Indicator:
        self._fps_label = QtWidgets.QLabel("FPS: 0")
        self._fps_label.setAlignment(QtCore.Qt.AlignCenter)
        self._fps_label.setStyleSheet("font-size: 20px; font-weight: bold")
        self._canvas.timer.connect(self._update_fps)

        layout.addWidget(self._start_stop_button)
        layout.addWidget(self._select_profile)
        layout.addWidget(self._seed_label)
        layout.addWidget(self._seed_input)
        layout.addWidget(self._num_flies_label)
        layout.addWidget(self._num_flies_input)
        layout.addWidget(self._cluster_boid_ratio_label)
        layout.addWidget(self._cluster_boid_ratio_slider)
        layout.addWidget(self._cluster_types_label)
        layout.addWidget(self._cluster_types_input)
        layout.addWidget(self._boid_types_label)
        layout.addWidget(self._boid_types_input)
        layout.addWidget(self._interaction_matrix_label)
        layout.addStretch(1)
        layout.addWidget(self._fps_label)
        layout.setSpacing(15)
        layout.setContentsMargins(10, 10, 10, 100)

        self.setLayout(layout)

    def _start_stop(self):
        if self._start_stop_button.text() == "Start":
            self._start_stop_button.setText("Stop")
            self._canvas.start()
        else:
            self._start_stop_button.setText("Start")
            self._fps_label.setText("FPS: 0")
            self._canvas.timer.stop()

    def _update_fps(self, event):
        self._fps_label.setText(f"FPS: {self._canvas.canvas.fps:.2f}")

    def _update_num_flies(self, event):
        pass

    def _update_cluster_boid_ratio(self, event):
        pass

    def _update_seed(self, event):
        pass

    def _update_profile(self, event):
        pass


if __name__ == "__main__":
    simulation = Simulation([100] * 2, seed=1234, limits=(100, 100, 100))
    simulation.update()  # dummy call to avoid frame freeze on first update

    visual = Visualiser()
    visual.set_simulation_instance(simulation)
    visual.draw_boundary()
    # visual.update('dummy')

    app = use_app("pyqt5")
    app.create()

    # visual.timer = app.timer()
    # visual.timer.connect(visual.update)

    window = MainWindow(visual)
    window.showMaximized()
    # window.showFullScreen()
    # window.show()
    app.run()

    print("Done")
