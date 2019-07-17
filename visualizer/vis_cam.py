from .vis_base import Visualizer_Cam_Base
import matplotlib.pyplot as plt
import numpy as np
from imageai.Detection import ObjectDetection
from sklearn.cluster import DBSCAN


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


class Visualizer_Cam_Data(Visualizer_Cam_Base):
    def create_fig(self):
        plt.ion()
        fig = plt.figure()
        ax0 = fig.add_subplot(111)
        plt.show()
        self.colors = [np.random.rand(3, ) for _ in range(20)]
        return ax0

    def plot(self, idx, fig, frame, runflag, detection=None):
        ax0 = fig

        # xs, ys, zs = np.split(frame.T, 3)
        ax0.cla()
        # ax0.plot(xs, ys, 'b.')
        ax0.set_xlim(self.xlim)
        ax0.set_ylim(self.ylim)
        ax0.set_xlabel('x (m)')
        ax0.set_ylabel('y (m)')

        clusters = cluster_DBSCAN(frame.T)

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

            left_bound = -AoV
            right_bound = AoV

            for c in clusters:
                cxs, cys, czs = np.split(c.T, 3)
                centroid = np.average(c, axis=0)
                leftmost = np.min(c, axis=0)
                rightmost = np.max(c, axis=0)
                a1 = np.arctan(centroid[0]/centroid[1])
                a2 = np.arctan(leftmost[0]/leftmost[1])
                a3 = np.arctan(rightmost[0]/rightmost[1])
                ax0.plot(cxs, cys, 'b.')

                label = 'None'

                for i, (x1, y1, x2, y2) in enumerate(detection):
                    cam_left = (x1 - self.cam_w/2)/(self.cam_w/2) * AoV
                    cam_right = (x2 - self.cam_w/2)/(self.cam_w/2) * AoV

                    # ignore if centre out of bound
                    if a1 < cam_left or a1 > cam_right:
                        continue

                    # ignore if centre not close to camera centre
                    if not close_to(a1, (cam_left + cam_right)/2):
                        continue

                    # true if two overlap by 80%
                    if close_to(a2, cam_left) and close_to(a3, cam_right) and (a3-a2)/(cam_right - cam_left) > 0.7:
                        ax0.plot(cxs, cys, 'r.')
                        label = 'True'
                        break

                # false if in aov and not fit into any camera box
                if label == 'None' and a2 > left_bound and a3 < right_bound:
                    if (a3-a2)/(cam_right - cam_left) < 0.5 or far_to(a2, cam_left) or far_to(a3, cam_right):
                        label = 'False'
                        ax0.plot(cxs, cys, 'y.')



        keyPressed = plt.waitforbuttonpress(timeout=0.005)
        if keyPressed:
            runflag.value = 0


def close_to(a, b):
    return abs(a-b) < 0.1


def far_to(a, b):
    return abs(a-b) > 0.3

def cluster_DBSCAN(data, min_points=20):
    # pdb.set_trace()
    if not data.any() or data.shape[0] < 10:
        return None
    model = DBSCAN(eps=0.06)
    model.fit((data[:, :2]))
    labels = model.labels_
    clusters = []

    for _, class_idx in enumerate(np.unique(labels)):
        if class_idx != -1:
            class_data = data[labels == class_idx]
            if class_data.shape[0] < min_points:
                continue
            clusters.append(class_data)

    return clusters
