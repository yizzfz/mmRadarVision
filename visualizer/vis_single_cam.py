from .vis_base import Visualizer_Cam_Base
import matplotlib.pyplot as plt
import numpy as np
from imageai.Detection import ObjectDetection



AoV = 30/180*np.pi

class Visualizer_Cam_Single(Visualizer_Cam_Base):
    def create_fig(self):
        plt.ion()
        fig = plt.figure()
        ax0 = fig.add_subplot(111)
        plt.show()
        self.colors = [np.random.rand(3, ) for _ in range(20)]
        return ax0

    def plot(self, idx, fig, frame, runflag, detection=None):
        ax0 = fig
        xs, ys, zs = np.split(frame.T, 3)
        ax0.cla()
        ax0.plot(xs, ys, 'b.')
        ax0.set_xlim(self.xlim)
        ax0.set_ylim(self.ylim)
        ax0.set_xlabel('x (m)')
        ax0.set_ylabel('y (m)')

        if detection is not None:
            for i, (x1, y1, x2, y2) in enumerate(detection):
                color = self.colors[i]
                cam_left = (x1 - self.cam_w/2)/(self.cam_w/2) * AoV
                cam_right = (x2 - self.cam_w/2)/(self.cam_w/2) * AoV
                for angle in [cam_left, cam_right]:
                    if angle < 0:
                        ax0.plot([0, self.xlim[0]], [
                            0, self.xlim[0]/np.tan(angle)], c=color)
                    elif angle > 0:
                        ax0.plot([0, self.xlim[1]], [
                            0, self.xlim[1]/np.tan(angle)], c=color)
                    else:
                        ax0.plot([0, 0], [0, self.ylim[1]], c=color)
                

        keyPressed = plt.waitforbuttonpress(timeout=0.005)
        if keyPressed:
            runflag.value = 0

    # def cam_process(self, frame):
    #     return frame, [(144, 288, 310, 457)]

