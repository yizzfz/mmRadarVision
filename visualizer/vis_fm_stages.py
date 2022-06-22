"""Plot data at each stage of FM, debugging purpose only"""
import matplotlib.pyplot as plt
import numpy as np
import traceback

class Visualizer_Single_FM_Stages():
    def __init__(self, queues, fm=None, xlim=[-2, 2], ylim=[0, 4], zlim=[0, 2], n_row=1, n_col=2):
        print('Warning: Non-standard visualizer')
        self.queues = queues
        self.xlim = xlim
        self.ylim = ylim
        self.zlim = zlim
        self.fm = fm
        self.n_row = n_row
        self.n_col = n_col
        self.n = n_row * n_col

    def run(self, runflag):
        fig = self.create_fig()
        while runflag.value == 1:
            try:
                for i, q in enumerate(self.queues):
                    if not q.empty():
                        frame = q.get(block=True, timeout=3)
                        if self.fm:
                            frames_to_draw = []
                            for f in self.fm:
                                frame = f.run(frame)
                                frames_to_draw.append(frame)
                        self.plot(i, fig, frames_to_draw, runflag)

            except Exception as e:
                print('Exception from visualization thread:', e)
                print(traceback.format_exc())
                runflag.value = 0
                break

    def create_fig(self):
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