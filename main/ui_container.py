from PyQt5 import QtWidgets, QtGui, QtCore
from vizman import Visualiser
from simulation import Simulation
from vispy.app import use_app

# TODO Timestamp Connection to Simulation
# TODO Interaction Matrix Colored Dots

# CANVAS_SIZE = (800, 600)
INIT_CONFIG = {
    "seed": 434,
    "boundary": "100, 100, 100",
    "clusters": "200, 200, 200, 200",
    "boids": "0, 0",
    "interactions": 4,
    "timestep": 0.1,
}


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

        self.interaction_i = 0
        self.interaction_j = 0

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
        self._seed_input.setValidator(QtGui.QIntValidator())
        self._seed_input.setText(str(INIT_CONFIG["seed"]))

        # # Number of flies input
        # self._num_flies_label = QtWidgets.QLabel("Number of Flies:")
        # self._num_flies_label.setAlignment(QtCore.Qt.AlignLeft)
        # # self._num_flies_label.setStyleSheet('font-size: 20px; font-weight: bold')
        # self._num_flies_input = QtWidgets.QLineEdit()
        # self._num_flies_input.setText("100, 100, 100")
        # self._num_flies_input.setValidator(QtGui.QIntValidator())
        # # self._num_flies_input.textChanged.connect(self._canvas.update_num_flies)

        # # Clusters to Boid Ratio Slider
        # self._cluster_boid_ratio_label = QtWidgets.QLabel("Cluster to Boid Ratio:")
        # self._cluster_boid_ratio_label.setAlignment(QtCore.Qt.AlignLeft)
        # # self._cluster_boid_ratio_label.setStyleSheet('font-size: 20px; font-weight: bold')
        # self._cluster_boid_ratio_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        # self._cluster_boid_ratio_slider.setMinimum(0)
        # self._cluster_boid_ratio_slider.setMaximum(20)
        # self._cluster_boid_ratio_slider.setValue(0)
        # self._cluster_boid_ratio_slider.setTickPosition(QtWidgets.QSlider.TicksBelow)
        # self._cluster_boid_ratio_slider.setTickInterval(5)
        # self._cluster_boid_ratio_slider.setSingleStep(5)
        # # self._cluster_boid_ratio_slider.valueChanged.connect(self._canvas.update_cluster_boid_ratio)

        # Boundary Limits X,Y,Z
        self._boundary_limits_label = QtWidgets.QLabel("Boundary Limits: ")
        self._boundary_limits_label.setAlignment(QtCore.Qt.AlignLeft)
        # self._boundary_limits_label.setStyleSheet('font-size: 20px; font-weight: bold')
        self._boundary_limits_input = QtWidgets.QLineEdit()
        self._boundary_limits_input.setText(str(INIT_CONFIG["boundary"]))
        # self._boundary_limits_input.textChanged.connect(self._canvas.update_boundary_limits)

        # Cluster Types:
        self._cluster_types_label = QtWidgets.QLabel("Cluster Types:")
        self._cluster_types_label.setAlignment(QtCore.Qt.AlignLeft)
        # self._cluster_types_label.setStyleSheet('font-size: 20px; font-weight: bold')
        self._cluster_types_input = QtWidgets.QLineEdit()
        self._cluster_types_input.setText(INIT_CONFIG["clusters"])
        # self._cluster_types_input.setValidator(QtGui.QIntValidator())
        # self._cluster_types_input.textChanged.connect(self._canvas.update_cluster_types)

        # Boid Types:
        self._boid_types_label = QtWidgets.QLabel("Boid Types:")
        self._boid_types_label.setAlignment(QtCore.Qt.AlignLeft)
        # self._boid_types_label.setStyleSheet('font-size: 20px; font-weight: bold')
        self._boid_types_input = QtWidgets.QLineEdit()
        self._boid_types_input.setPlaceholderText(INIT_CONFIG["boids"])
        self._boid_types_input.setValidator(QtGui.QIntValidator())
        # self._boid_types_input.textChanged.connect(self._canvas.update_boid_types)

        # Interaction Matrix:
        self._interaction_matrix_label = QtWidgets.QLabel("Interaction Matrix:")
        self._interaction_matrix_label.setAlignment(QtCore.Qt.AlignLeft)
        # self._interaction_matrix_label.setStyleSheet('font-size: 20px; font-weight: bold')

        # Interaction Matrix Display
        # grid of colored square buttons
        self._interaction_matrix = QtWidgets.QGridLayout()
        self._interaction_matrix.setSpacing(5)
        self._interaction_matrix_buttons = []
        self.COLOUR_LIST = self._canvas.COLOUR_LIST 
        for i in range(INIT_CONFIG["interactions"] + 1):
            for j in range(INIT_CONFIG["interactions"] + 1):
                if i>0 and j>0:
                    button = QtWidgets.QPushButton()
                    button.setFixedSize(30, 30)
                    button.setStyleSheet("background-color: rgb(0, 0, 0);")
                    button.clicked.connect(
                        lambda nill, i=i, j=j: self._update_specific_interaction(i-1, j-1)
                    )
                    # print(f"i: {i}, j: {j}")
                elif i==0 and j>0:

                    button = QtWidgets.QPushButton()
                    button.setFixedSize(20, 20)
                    button.setStyleSheet(f"background-color: {self.COLOUR_LIST[j-1]};")
                    # print(f"2")
                elif j == 0 and i>0:

                    button = QtWidgets.QPushButton()
                    button.setFixedSize(20, 20)
                    button.setStyleSheet(f"background-color: {self.COLOUR_LIST[i-1]};")
                    # print(f"3")
                else:
                    button = QtWidgets.QPushButton()
                    button.setFixedSize(10, 10)
                    button.setStyleSheet("background-color: rgb(0, 0, 0);")
                    # print(f"asd")
                # align buttons in interaction matrix
                self._interaction_matrix.addWidget(button, i, j, QtCore.Qt.AlignCenter)
                self._interaction_matrix_buttons.append((button, i, j))

        # 4 Params:
        self._params_label = QtWidgets.QLabel(
            f"Parameters: {self.interaction_i}, {self.interaction_j}"
        )
        self._params_label.setAlignment(QtCore.Qt.AlignLeft)
        # self._params_label.setStyleSheet('font-size: 20px; font-weight: bold')

        # 4 Params Display
        self._params = QtWidgets.QVBoxLayout()
        self._params.setSpacing(5)
        self._params_inputs = []
        for i in range(4):
            input = QtWidgets.QLineEdit()
            input.setPlaceholderText("param" + str(i))
            input.setValidator(QtGui.QDoubleValidator())
            # input.textChanged.connect(self._canvas.update_params)
            self._params.addWidget(input)
            self._params_inputs.append(input)

        # Update Params Button:
        self._update_params_button = QtWidgets.QPushButton("Update Live")
        self._update_params_button.clicked.connect(self._update_params_button_clicked)
        
        # Timestep:
        self._timestep_label = QtWidgets.QLabel("Timestep:")
        self._timestep_label.setAlignment(QtCore.Qt.AlignLeft)
        # self._timestep_label.setStyleSheet('font-size: 20px; font-weight: bold')
        self._timestep_input = QtWidgets.QLineEdit()
        self._timestep_input.setText(str(INIT_CONFIG["timestep"]))
        self._timestep_input.setValidator(QtGui.QDoubleValidator())
        # self._timestep_input.textChanged.connect(self._canvas.update_timestep)


        # Update Button:
        self._update_button = QtWidgets.QPushButton("Update")
        self._update_button.clicked.connect(self._update_button_clicked)

        # FPS Indicator:
        self._fps_label = QtWidgets.QLabel("FPS: 0")
        self._fps_label.setAlignment(QtCore.Qt.AlignCenter)
        self._fps_label.setStyleSheet("font-size: 20px; font-weight: bold")
        self._canvas.timer.connect(self._update_fps)

        # Layout:
        layout.addWidget(self._select_profile)
        layout.addWidget(self._seed_label)
        layout.addWidget(self._seed_input)
        # layout.addWidget(self._num_flies_label)
        # layout.addWidget(self._num_flies_input)
        # layout.addWidget(self._cluster_boid_ratio_label)
        # layout.addWidget(self._cluster_boid_ratio_slider)
        layout.addWidget(self._cluster_types_label)
        layout.addWidget(self._cluster_types_input)
        layout.addWidget(self._boid_types_label)
        layout.addWidget(self._boid_types_input)
        layout.addWidget(self._boundary_limits_label)
        layout.addWidget(self._boundary_limits_input)
        layout.addWidget(self._interaction_matrix_label)
        layout.addLayout(self._interaction_matrix)
        layout.addWidget(self._params_label)
        layout.addLayout(self._params)
        layout.addWidget(self._update_params_button)
        layout.addWidget(self._timestep_label)
        layout.addWidget(self._timestep_input)
        layout.addWidget(self._update_button)
        layout.addWidget(self._start_stop_button)
        layout.addStretch(1)
        layout.addWidget(self._fps_label)
        layout.setSpacing(15)
        layout.setContentsMargins(10, 10, 10, 50)

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

    def _redraw_interaction_matrix(self, n):
        self._remove_interaction_matrix_buttons()
        self._interaction_matrix_buttons = []

        for i in range(n + 1):
            for j in range(n + 1):
                if i>0 and j>0:
                    button = QtWidgets.QPushButton()
                    button.setFixedSize(30, 30)
                    if self._canvas.simulation.parameter_matrix[3, i-1, j-1] > 0:
                        button.setStyleSheet("background-color: rgb(50, 255, 50);")
                    else:
                        button.setStyleSheet("background-color: rgb(255, 50, 50);")

                    button.clicked.connect(
                        lambda nill, i=i, j=j: self._update_specific_interaction(i-1, j-1)
                    )
                    # print(f"i: {i}, j: {j}")
                elif i==0 and j>0:

                    button = QtWidgets.QPushButton()
                    button.setFixedSize(20, 20)
                    button.setStyleSheet(f"background-color: {self.COLOUR_LIST[j-1]};")
                    # print(f"2")
                elif j == 0 and i>0:

                    button = QtWidgets.QPushButton()
                    button.setFixedSize(20, 20)
                    button.setStyleSheet(f"background-color: {self.COLOUR_LIST[i-1]};")
                    # print(f"3")
                else:
                    button = QtWidgets.QPushButton()
                    button.setFixedSize(10, 10)
                    button.setStyleSheet("background-color: rgb(0, 0, 0);")
                    # print(f"asd")
                self._interaction_matrix.addWidget(button, i, j, QtCore.Qt.AlignCenter)
                self._interaction_matrix_buttons.append((button, i, j))

    def _remove_interaction_matrix_buttons(self):
        for i in reversed(range(self._interaction_matrix.count())):
            self._interaction_matrix.itemAt(i).widget().setParent(None)

    def _update_specific_interaction(self, i, j):
        self._params_label.setText(f"{self.COLOUR_LIST[i]} to {self.COLOUR_LIST[j]}")
        self.interaction_i = i
        self.interaction_j = j
        self._display_params()

    def _display_params(self):
        if self._canvas.simulation is not None:
            parameter_matrix = self._canvas.simulation.parameter_matrix
            relevant_params = parameter_matrix[
                :, self.interaction_i, self.interaction_j
            ]
            for i in range(4):
                self._params_inputs[i].setText(f"{relevant_params[i]:.2f}")

    def _update_params_button_clicked(self, event):
        print("Update Params Button Clicked")
        new_params = [float(i.text()) for i in self._params_inputs]
        print("New Params:", new_params)
        self._canvas.simulation.update_parameter_matrix(
            self.interaction_i, self.interaction_j, new_params
        )
        clusters = [int(i) for i in self._cluster_types_input.text().split(",")]
        self._redraw_interaction_matrix(len(clusters))
        self._redraw_boundaries(event)

    def _redraw_boundaries(self, event):
        limits = [int(i) for i in self._boundary_limits_input.text().split(",")]
        self._canvas.draw_boundary(limits)


    def _update_button_clicked(self, event):
        seed = int(self._seed_input.text())
        clusters = [int(i) for i in self._cluster_types_input.text().split(",")]
        # boids = [int(i) for i in self._boid_types_input.text().split(",")]
        limits = [int(i) for i in self._boundary_limits_input.text().split(",")]

        print("Update Button Clicked")
        print("Seed:", seed)
        print("Clusters:", clusters)
        # print("Boids:", boids)
        print("Limits:", limits)

        new_simulation = Simulation(clusters, limits=limits, seed=seed)
        # new_simulation.update()
        self._canvas.set_simulation_instance(new_simulation)
        self._canvas.draw_boundary()

        self._redraw_interaction_matrix(len(clusters))
        self._display_params()


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
