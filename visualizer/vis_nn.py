from .vis_base import Visualizer_NN_Base
import matplotlib.pyplot as plt
import numpy as np
import pickle
import sys
sys.path.append("..")
from util import cluster_DBSCAN, frame_to_mat


class Visualizer_NN(Visualizer_NN_Base):
    def create_fig(self):
        plt.ion()
        fig = plt.figure()
        ax0 = fig.add_subplot(111)
        plt.show()
        return ax0

    def plot(self, idx, fig, frame, centroids, labels, runflag):
        ax0 = fig
        ax0.cla()
        ax0.set_xlim(self.xlim)
        ax0.set_ylim(self.ylim)
        ax0.set_xlabel('x (m)')
        ax0.set_ylabel('y (m)')
        xs, ys, zs = np.split(frame.T, 3)
        ax0.plot(xs, ys, '.b', alpha=0.2)


        for centroid, label in zip(centroids, labels):
            if label == 0:
                circle = plt.Circle((centroid[0], centroid[1]), 0.1, color='g', alpha=1)
            # else:
            #     circle = plt.Circle((centroid[0], centroid[1]), 0.1, color='r')
                ax0.add_artist(circle)


        keyPressed = plt.waitforbuttonpress(timeout=0.005)
        if keyPressed:
            runflag.value = 0

    def prepare_data(self, frame): 
        res = frame_to_mat(frame, ret_centroids=True)
        if res is None:
            return None, None

        return res