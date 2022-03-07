from .vis_base import Visualizer_Base
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
import numpy as np
from matplotlib import cm


class Visualizer_Single_3D(Visualizer_Base):
    """Plot x-y points and x-y-z (3D) points."""
    def create_fig(self):
        plt.ion()
        fig0 = plt.figure()
        ax0 = fig0.add_subplot(121)
        ax1 = fig0.add_subplot(122, projection='3d')

        ls0, = ax0.plot([], [], '.')

        ax0.set_xlim(self.xlim)
        ax0.set_ylim(self.ylim)
        ax0.set_xlabel('x (m)')
        ax0.set_ylabel('y (m)')

        plt.show()
        self.fig0 = ls0
        self.fig1 = ax1

    def plot_combined(self, frame, runflag):
        frame = frame[frame[:, 0].argsort()]
        xs, ys, zs = np.split(frame.T, 3)

        ls0 = self.fig0
        ls0.set_xdata(xs)
        ls0.set_ydata(ys)

        ax1 = self.fig1
        ax1.cla()
        ax1.set_xlim(self.xlim)
        ax1.set_ylim(self.ylim)
        ax1.set_xlabel('x (m)')
        ax1.set_ylabel('y (m)')
        ax1.set_zlabel('Height (m)')

        ax1.set_zlim([-1, 1])

        self._plot3D_lines(ax1, frame)
        keyPressed = plt.waitforbuttonpress(timeout=0.005)
        if keyPressed:
            runflag.value = 0

    def _plot3D(self, ax, frame):
        xs, ys, zs = np.split(frame.T, 3)
        if xs.shape[1] > 3:
            ax.plot_trisurf(xs.flatten(), ys.flatten(), zs.flatten())

    def _plot3D_lines(self, ax, frame):
        for (x, y, z) in frame:
            ax.plot([x, x], [y, y], [-1, z], color='blue')
            ax.plot([0, x], [0, y], [-1, -1], color='red', alpha=0.2)

    def _plot3D_complex(self, ax, frame):
        frame = np.round(frame, 2)
        xs, ys, zs = np.split(frame.T, 3)

        xg, yg = np.mgrid[-2:2:0.01, 0:2:0.01]
        zg = np.zeros(xg.shape)-1
        
        for (x, y, z) in frame:
            zg[(xg==x) & (yg==y)] = z

        print((zg!=-1).sum())

        ax.plot_surface(xg, yg, zg, cmap=cm.Blues)
