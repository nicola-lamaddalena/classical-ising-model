import sys
import numpy as np
from PyQt6.QtWidgets import QApplication, QHBoxLayout, QMainWindow, QWidget, QVBoxLayout, QSlider, QPushButton
from PyQt6.QtCore import QTimer, Qt
import pyqtgraph as pg
from metro import metropolis
from utils import energy, magnetization

MAIN_COLOR = "#191923"
BKG_COLOR = "#E9EEF3"
pg.setConfigOption('background', BKG_COLOR)
pg.setConfigOption('foreground', MAIN_COLOR)

class IsingMVP(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Modello di Ising 2D classico - Temperatura critica")
        self.resize(1024, 768) 
        self.DOWN_COLOR = (200, 62, 62)
        self.UP_COLOR = (59, 96, 228)
        self.N = 512
        self.T = 2.269
        self.acc = 1
        self.h = 0.0
        self.lattice = np.random.choice([-1, 1], size=(self.N, self.N))
        
        self.even_mask = np.zeros((self.N, self.N), dtype=bool)
        self.even_mask[::2, ::2] = True
        self.even_mask[1::2, 1::2] = True

        self.hist_limit = 200
        self.eng_hist = []
        self.magn_hist = []

        # layout verticale
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        hbox = QHBoxLayout()
        main_widget.setLayout(hbox)
        
        # slider della temperatura 
        self.temp_slider = QSlider(Qt.Orientation.Horizontal)
        self.temp_slider.setMinimum(100)   # T = 1.0
        self.temp_slider.setMaximum(500)   # T = 5.0
        self.temp_slider.setValue(227)     # T = 2.27
        self.temp_slider.valueChanged.connect(self.update_temperature)

        self.tc_button = QPushButton("Temperatura Critica (Tc = 2.269)")
        self.tc_button.setStyleSheet(f"background-color: {BKG_COLOR}; color: {MAIN_COLOR}; font-weight: bold; padding: 5px;")
        self.tc_button.clicked.connect(self.reset_to_critical_temp)

        self.image_view = pg.ImageView()
        self.image_view.ui.histogram.hide()
        self.image_view.ui.roiBtn.hide()
        self.image_view.ui.menuBtn.hide()
        hbox.addWidget(self.image_view, stretch=2)
    
        vbox_right = QVBoxLayout()
        hbox.addLayout(vbox_right, stretch=1)

        self.mag_plot = pg.PlotWidget(title="Magnetizzazione media per spin")
        self.mag_plot.setYRange(-1.1, 1.1)
        self.mag_curve = self.mag_plot.plot(pen=self.UP_COLOR)
        vbox_right.addWidget(self.mag_plot)
        
        self.energy_plot = pg.PlotWidget(title="Energia media per spin")
        self.energy_plot.setYRange(-2.1, 0.1)
        self.energy_curve = self.energy_plot.plot(pen=self.UP_COLOR)
        vbox_right.addWidget(self.energy_plot)

        vbox_right.addWidget(self.temp_slider)
        vbox_right.addWidget(self.tc_button)

        posizioni = np.array([0.0, 1.0])
        colori = np.array([self.DOWN_COLOR, self.UP_COLOR], dtype=np.ubyte)
        cmap = pg.ColorMap(posizioni, colori)
        self.image_view.setColorMap(cmap)
        self.image_view.setLevels(-1, 1)
        
        self.image_view.ui.histogram.hide()
        self.image_view.ui.roiBtn.hide()
        self.image_view.ui.menuBtn.hide()

        self.timer = QTimer()
        self.timer.timeout.connect(self.simulation_step)
        self.timer.start(20) # Esegue il loop ogni 20 millisecondi (~50 FPS)

    def update_temperature(self, value):
        self.T = value / 100.0
        self.statusBar().showMessage(f"Temperatura corrente: {self.T:.2f}")
    
    def update_acc(self, value):
        self.acc = value
        self.statusBar().showMessage(f"Accoppiamento corrente: {self.acc}")

    def simulation_step(self):
        metropolis(self.lattice, self.N, J=self.acc, h=self.h, T=self.T, even_mask=self.even_mask)
        self.image_view.setImage(self.lattice, autoLevels=False)

        m = magnetization(self.lattice)
        e = energy(self.lattice, self.acc)
        self.magn_hist.append(m)
        self.eng_hist.append(e)

        if len(self.magn_hist) > self.hist_limit:
            self.magn_hist.pop(0)
            self.eng_hist.pop(0)

        self.mag_curve.setData(self.magn_hist)
        self.energy_curve.setData(self.eng_hist)


    def reset_to_critical_temp(self):
        """Riporta istantaneamente lo slider e la variabile T al valore critico."""
        self.temp_slider.setValue(227)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = IsingMVP()
    window.show()
    sys.exit(app.exec())
