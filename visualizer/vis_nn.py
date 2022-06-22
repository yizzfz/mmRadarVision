"""Code in this file are still under development"""
from .vis_base import Visualizer_Base
import matplotlib.pyplot as plt
import numpy as np
import pickle
import sys
sys.path.append("..")
from util import cluster_DBSCAN, frame_to_mat

class Visualizer_NN_Base(Visualizer_Base):
    """Use a pre-trained neural network model to do some stuff"""
    def __init__(self, model, *args, **kwargs):
        self.model = model
        super().__init__(*args, **kwargs)

    def prepare_data(self, *args):
        raise NotImplementedError

class Visualizer_NN(Visualizer_NN_Base):
    def plot_combined(self, frame, runflag):
        mat, centroids = self.prepare_data(frame)
        if mat is None:
            return
        labels = self.model.model_predict(mat)
        ax0 = self.ax0
        ax0.cla()
        ax0.set_xlim(self.xlim)
        ax0.set_ylim(self.ylim)
        ax0.set_xlabel('x (m)')
        ax0.set_ylabel('y (m)')
        xs, ys, zs = np.split(frame.T, 3)
        ax0.plot(xs, ys, '.b', alpha=0.2)

        for centroid, label in zip(centroids, labels):
            # if label == 0:
                circle = plt.Circle((centroid[0], centroid[1]), 0.1, color='g', alpha=1)
            # else:
            #     circle = plt.Circle((centroid[0], centroid[1]), 0.1, color='r')
                ax0.add_artist(circle)
                ax0.text(centroid[0], centroid[1], str(label))

        keyPressed = plt.waitforbuttonpress(timeout=0.005)
        if keyPressed:
            runflag.value = 0

    def create_fig(self):
        plt.ion()
        fig = plt.figure()
        self.ax0 = fig.add_subplot(111)
        plt.show()

    def prepare_data(self, frame): 
        res = frame_to_mat(frame, ret_centroids=True)
        if res is None:
            return None, None
        return res