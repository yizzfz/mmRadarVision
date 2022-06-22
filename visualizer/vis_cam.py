"""Code in this file is not guaranteed to work"""
from .vis_base import Visualizer_Base
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append("..")
from util import cluster_DBSCAN

AoV = 40/180*np.pi

class Visualizer_Cam(Visualizer_Base):
    """Use a camera detector to filter radar data"""
    def create_fig(self):
        plt.ion()
        fig = plt.figure()
        ax0 = fig.add_subplot(111)
        plt.show()
        self.colors = [np.random.rand(3, ) for _ in range(20)]
        self.ax0 = ax0

    def plot_combined(self, frame, runflag):
        _, detection = self.cam.get_detection()
        ax0 = self.ax0
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


class Visualizer_Cam_Data_Generator(Visualizer_Cam):
    """Generate labelled data based a camera dedector"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.keys = ['True', 'False']
        self.data = {k: [] for k in self.keys}

    def plot_combined(self, frame, runflag):
        _, detection = self.cam.get_detection()
        xs, ys, zs = np.split(frame.T, 3)
        ax0 = self.ax0
        ax0.cla()
        # ax0.plot(xs, ys, '.k')
        ax0.set_xlim(self.xlim)
        ax0.set_ylim(self.ylim)
        ax0.set_xlabel('x (m)')
        ax0.set_ylabel('y (m)')

        clusters = cluster_DBSCAN(frame)
        if clusters is None:
            return

        if detection is not None:
            for i, (x1, y1, x2, y2) in enumerate(detection):
                if y2/self.cam_h < 0.9:
                    continue
                color = self.colors[i]
                cam_left = (x1 - self.cam_w/2)/(self.cam_w/2) * AoV
                cam_right = (x2 - self.cam_w/2)/(self.cam_w/2) * AoV
                for angle in [cam_left, cam_right]:
                    if angle < 0:
                        ax0.plot([0, self.xlim[0]], [
                            0-0.2, self.xlim[0]/np.tan(angle)-0.2], c=color)
                    elif angle > 0:
                        ax0.plot([0, self.xlim[1]], [
                            0-0.2, self.xlim[1]/np.tan(angle)-0.2], c=color)
                    else:
                        ax0.plot([0, 0], [0, self.ylim[1]], c=color)

            left_bound = -AoV
            right_bound = AoV

            for c in clusters:
                cxs, cys, czs = np.split(c.T, 3)
                centroid = np.average(c, axis=0)

                a1 = np.arctan(centroid[0]/centroid[1])
                a2 = np.min(cxs/cys)
                a3 = np.max(cxs/cys)
                ax0.plot(cxs, cys, '.b')

                label = 'None'
                if centroid[1] < 0.5 or centroid[1] > 2:
                    continue

                for i, (x1, y1, x2, y2) in enumerate(detection):
                    if y2/self.cam_h < 0.9:
                        continue
                    cam_left = (x1 - self.cam_w/2)/(self.cam_w/2) * AoV
                    cam_right = (x2 - self.cam_w/2)/(self.cam_w/2) * AoV

                    # ignore if centre out of bound
                    if a1 < cam_left or a1 > cam_right:
                        continue

                    # ignore if centre not close to camera centre
                    if not close_to(a1, (cam_left + cam_right)/2):
                        continue

                    # if not far_to(a1, (cam_left + cam_right)/2) or (not far_to(a2, cam_left) and not far_to(a3, cam_right)):
                    #     label = 'Discard'

                    # true if two overlap by some threshold
                    if min(cam_right, a3) - max(cam_left, a2) > (cam_right - cam_left) * 0.2:
                        ax0.plot(cxs, cys, '.g')
                        # print(cam_left, cam_right)
                        # print(a2, a3)
                        # plt.waitforbuttonpress()
                        label = 'True'
                        break

                # false if in AoV and not fit into any camera box
                if label == 'None':
                    label = 'False'
                    ax0.plot(cxs, cys, '.r')


                if self.save and label in self.keys:
                    self.data[label].append(c)


        keyPressed = plt.waitforbuttonpress(timeout=0.005)
        if keyPressed:
            runflag.value = 0


def close_to(a, b):
    return abs(a-b) < 0.2

def far_to(a, b):
    return abs(a-b) > 0.5

