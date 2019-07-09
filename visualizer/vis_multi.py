from .vis_base import Visualizer_Multi_Base
import matplotlib.pyplot as plt
import numpy as np


class Visualizer_Multi(Visualizer_Multi_Base):
    def create_fig(self):
        if self.n_row == 1:
            return self.create_fig_one_row()
        else:
            return self.create_fig_n_row()

    def create_fig_one_row(self):
        plt.ion()
        fig, axs = plt.subplots(1, self.n_col)
        ls = []
        for j in range(self.n_col):
            ax = axs[j]
            ax.set_xlim(self.xlim)
            ax.set_ylim(self.ylim)
            ax.set_xlabel('x (m)')
            ax.set_ylabel('y (m)')
            l, = ax.plot([], [], '.')
            ls.append(l)

        plt.tight_layout()
        plt.show()
        return ls

    def create_fig_n_row(self):
        plt.ion()
        fig, axs = plt.subplots(self.n_row, self.n_col)
        ls = []
        for i in range(self.n_row):
            for j in range(self.n_col):
                ax = axs[i, j]
                ax.set_xlim(self.xlim)
                ax.set_ylim(self.ylim)
                ax.set_xlabel('x (m)')
                ax.set_ylabel('y (m)')
                l, = ax.plot([], [], '.')
                ls.append(l)

        plt.tight_layout()
        plt.show()
        return ls

    def plot(self, idx, fig, frames, runflag):
        for i, l in enumerate(fig):
            if len(frames) <= i:
                frame = frames[-1]
            else:
                frame = frames[i]
            if frame is None:
                frame = np.asarray([[0, 0, 0]])
            xs, ys, zs = np.split(frame.T, 3)
            l.set_xdata(xs)
            l.set_ydata(ys)
        keyPressed = plt.waitforbuttonpress(timeout=0.005)
        if keyPressed:
            runflag.value = 0



