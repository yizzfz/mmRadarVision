from .vis_base import Visualizer_Base
import matplotlib.pyplot as plt
import numpy as np
import pickle


class Visualizer_Single(Visualizer_Base):
    def create_fig(self):
        plt.ion()
        fig = plt.figure()
        ax0 = fig.add_subplot(111)

        ls0, = ax0.plot([], [], '.')

        ax0.set_xlim(self.xlim)
        ax0.set_ylim(self.ylim)
        ax0.set_xlabel('x (m)')
        ax0.set_ylabel('y (m)')
        plt.show()
        return ls0

    def plot(self, idx, fig, frame, runflag):
        xs, ys, zs = np.split(frame.T, 3)
        fig.set_xdata(xs)
        fig.set_ydata(ys)
        keyPressed = plt.waitforbuttonpress(timeout=0.005)
        if keyPressed:
            runflag.value = 0


class Visualizer_Single_Height(Visualizer_Base):
    def create_fig(self):
        start = -1
        end = 1
        grid_h = 4
        grid_w = 7
        step = grid_w*grid_h
        step_height = (end-start)/step

        fig, axs = plt.subplots(grid_h, grid_w)

        ls = []
        plt.ion()

        h = start
        for i in range(grid_h):
            for j in range(grid_w):
                l, = axs[i, j].plot([], [], '.')
                ls.append(l)
                axs[i, j].set_xlim([-2, 2])
                axs[i, j].set_ylim([0, 4])
                title = f'{h:.3f} to {(h+step_height):.3f}'
                axs[i, j].title.set_text(title)
                h += step_height
        plt.show()
        return ls

    def plot(self, idx, fig, frame, runflag):
        start = -1
        end = 1
        grid_h = 4
        grid_w = 7
        step = grid_w*grid_h
        step_height = (end-start)/step

        ls = fig
        h = start
        for i in range(len(ls)):
            filtered = frame[(
                frame[:, 2] < h+step_height) & (frame[:, 2] >= h)]
            ls[i].set_xdata(filtered[:, 0])
            ls[i].set_ydata(filtered[:, 1])
            h += step_height

        keyPressed = plt.waitforbuttonpress(timeout=0.005)
        if keyPressed:
            runflag.value = 0


# record everything for a period of time
class Visualizer_Background(Visualizer_Base):
    def run(self, runflag):
        radar_frame = np.ndarray((0, 3))
        step = 0
        while runflag.value == 1:
            try:
                for i, q in enumerate(self.queues):
                    if not q.empty():
                        frame = q.get(block=True, timeout=3)
                        radar_frame = np.concatenate((radar_frame, frame))
                        step += 1

                if step > 80000:
                    runflag.value = 0
                    break


            except Exception as e:
                print('Exception from visualization thread:', e)
                print(traceback.format_exc())
                runflag.value = 0
                break

        if self.save:
            with open(self.save, 'wb') as f:
                pickle.dump(radar_frame, f)
