from .vis_base import Visualizer_Base
import matplotlib.pyplot as plt
import numpy as np

class Visualizer_Motor(Visualizer_Base):
    def __init__(self, *args, motorpos=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.motorpos = motorpos

    def plot_combined(self, frame, runflag):
        pos = self.motorpos.value/10000
        frame[:, 1] = frame[:, 1] + pos
        super().plot_combined(frame, runflag)

class Visualizer_AE(Visualizer_Base):
    """Plot x-y and x-z points.
    """
    def create_fig(self):
        fig, (ax0, ax1) = plt.subplots(1, 2)
        
        ax0.set_xlim(self.xlim)
        ax0.set_ylim(self.ylim)
        ax0.set_xlabel('x (m)')
        ax0.set_ylabel('depth (m)')

        ax1.set_xlim(self.xlim)
        ax1.set_ylim(self.zlim)
        ax1.set_xlabel('x (m)')
        ax1.set_ylabel('height (m)')

        self.ls0, = ax0.plot([], [], 'r.')   # r for red
        self.ls1, = ax1.plot([], [], 'r.')   # r for red

        self.ax0 = ax0
        self.ax1 = ax1

        plt.ion()
        plt.show()

    def plot_combined(self, frame, runflag):
        xs1, ys1, zs1 = np.squeeze(np.split(frame.T, 3))
        self.ls0.set_xdata(xs1)
        self.ls0.set_ydata(ys1)
        self.ls1.set_xdata(xs1)
        self.ls1.set_ydata(zs1)

        keyPressed = plt.waitforbuttonpress(timeout=0.005)
        if keyPressed:
            runflag.value = 0