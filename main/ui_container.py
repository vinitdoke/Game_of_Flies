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
        self._controls = ControlsWidget(_canvas = self._canvas)

        main_layout.addWidget(self._controls, 1)
        main_layout.addWidget(self._canvas.canvas.native, 10)

        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        # COSMETIC
        self.setWindowTitle('Game of Flies')
        self.setWindowIcon(QtGui.QIcon('icon.png'))


class ControlsWidget(QtWidgets.QWidget):

    def __init__(self, parent = None, _canvas = None):
        super().__init__(parent)

        self._canvas = _canvas
        layout = QtWidgets.QVBoxLayout()

        # Start/Stop Button
        self._start_stop_button = QtWidgets.QPushButton('Start')
        self._start_stop_button.clicked.connect(self._start_stop)
        layout.addWidget(self._start_stop_button)

        # Profile Selector:
        self._select_profile = QtWidgets.QComboBox()
        self._select_profile.addItems(['Clusters', 'Boids', 'More'])
        layout.addWidget(self._select_profile)

        # FPS Indicator:
        self._fps_label = QtWidgets.QLabel('FPS: 0')
        self._fps_label.setAlignment(QtCore.Qt.AlignCenter)
        self._fps_label.setStyleSheet('font-size: 20px; font-weight: bold')
        layout.addWidget(self._fps_label)
        self._canvas.timer.connect(self._update_fps)

        # Parameters:

        ## STATS
        # SEED
        # self._seed_label = QtWidgets.QLabel('Seed:')
        # self._seed_label.setAlignment(QtCore.Qt.AlignCenter)
        # self._seed_label.setStyleSheet('font-size: 20px; font-weight: bold')
        # layout.addWidget(self._seed_label)


        self.setLayout(layout)
    

    def _start_stop(self):
        if self._start_stop_button.text() == 'Start':
            self._start_stop_button.setText('Stop')
            self._canvas.start()
        else:
            self._start_stop_button.setText('Start')
            self._fps_label.setText('FPS: 0')
            self._canvas.timer.stop()

    def _update_fps(self, event):
        self._fps_label.setText(f'FPS: {self._canvas.canvas.fps:.2f}')


if __name__ == "__main__":

    simulation = Simulation([100]*9, seed = 1234, limits=(100, 100, 100))
    simulation.update()  # dummy call to avoid frame freeze on first update

    visual = Visualiser()
    visual.set_simulation_instance(simulation)
    visual.draw_boundary()
    # visual.update('dummy')

    app = use_app('pyqt5')
    app.create()

    # visual.timer = app.timer()
    # visual.timer.connect(visual.update)

    window = MainWindow(visual)
    window.showMaximized()
    # window.showFullScreen()
    # window.show()
    app.run()

    print('Done')