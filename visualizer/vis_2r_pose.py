"""Two radars are placed as a vertical array for posture capturing
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
import traceback
import datetime
import pickle
from .util_visualizer import *
from .vis_base import Visualizer_Base
from scipy import stats


class Visualizer_TwoR_Vertical(Visualizer_Base):
    def __init__(self, *args, plot_all_points=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.plot_all_points = plot_all_points

    def create_fig(self):
        fig0 = plt.figure(0)
        ax0 = fig0.add_subplot(121)
        ax1 = fig0.add_subplot(122)

        ax0.set_xlim(self.xlim)
        ax0.set_ylim(self.ylim)
        ax0.set_xlabel('x (m)')
        ax0.set_ylabel('depth (m)')

        ax1.set_xlim(self.xlim)
        ax1.set_ylim(self.zlim)
        ax1.set_xlabel('x (m)')
        ax1.set_ylabel('height (m)')

        radar1 = plt.Circle((0 - d_hor/2, 0), 0.05, color='r')
        radar2 = plt.Circle((0 + d_hor/2, 0), 0.05, color='b')
        ax0.add_artist(radar1)
        ax0.add_artist(radar2)

        self.ls0a, = ax0.plot([], [], 'r.')   # r for red
        self.ls0b, = ax0.plot([], [], 'bx')   # b for blue
        self.ls1a, = ax1.plot([], [], 'r.')   # r for red
        self.ls1b, = ax1.plot([], [], 'bx')   # b for blue

        self.ax0 = ax0
        self.ax1 = ax1

        plt.ion()
        plt.show()

    def plot_each(self, idx, frame, runflag):
        if frame is None:
            return
        if idx == 0:
            xs1, ys1, zs1 = np.split(frame.T, 3)
            xs1 = xs1.flatten()
            ys1 = ys1.flatten()
            zs1 = zs1.flatten()
            xs1 = xs1 - d_hor/2
            zs1 = zs1 + d_ver/2
            zs1 = zs1 + self.height1
            if self.plot_all_points:
                self.ls0a.set_xdata(xs1)
                self.ls0a.set_ydata(ys1)
                self.ls1a.set_xdata(xs1)
                self.ls1a.set_ydata(zs1)
            self.clusters1 = cluster_xyz(xs1, ys1, zs1)
            self.ps1 = xs1, ys1, zs1
        elif idx == 1:
            xs2, ys2, zs2 = np.split(frame.T, 3)
            xs2 = xs2.flatten()
            ys2 = ys2.flatten()
            zs2 = zs2.flatten()
            xs2 = xs2 + d_hor/2
            zs2 = zs2 - d_ver/2
            zs2 = zs2 + self.height2
            if self.plot_all_points:
                self.ls0b.set_xdata(xs2)
                self.ls0b.set_ydata(ys2)
                self.ls1b.set_xdata(xs2)
                self.ls1b.set_ydata(zs2)
            self.clusters2 = cluster_xyz(xs2, ys2, zs2)
            self.ps2 = xs2, ys2, zs2
        else:
            print('Error: Using more than two queues, but visualizer designed for two')
            runflag.value = 0