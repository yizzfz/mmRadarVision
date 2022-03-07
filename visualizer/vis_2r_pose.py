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

        radar1 = plt.Circle((0 - dp_hor/2, 0), 0.05, color='r')
        radar2 = plt.Circle((0 + dp_hor/2, 0), 0.05, color='b')
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
            xs1 = xs1 - dp_hor/2
            zs1 = zs1 + dp_ver/2
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
            xs2 = xs2 + dp_hor/2
            zs2 = zs2 - dp_ver/2
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

class Visualizer_TwoR_Vertical_9S(Visualizer_TwoR_Vertical):
    """attempt to draw posture using 9-segment approach"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        cv2.namedWindow('1', 0)

    def plot_combined(self, frame, runflag):
        xs1, ys1, zs1 = self.ps1
        xs2, ys2, zs2 = self.ps2
        xs1 = np.append(xs1, xs2)
        ys1 = np.append(ys1, ys2)
        zs1 = np.append(zs1, zs2)
        frame = np.asarray((xs1, ys1, zs1)).T

        posture_code = 0
        if frame.size > 30:
            posture_code = compute_posture_code(self.ps1, self.ps2)
        screen = draw_posture(posture_code)
        #     sz = 9
        #     screen = np.zeros((sz+2, sz+2))
        #     xlo = -0.5
        #     xhi = 0.5
        #     zlo = 0
        #     zhi = 2

        #     frame = np.append(ps1, ps2, axis=0)

        #     for x,_,z in frame:
        #         x = int((x-xlo)/(xhi-xlo)*sz)
        #         z = int(sz-(z-zlo)/(zhi-zlo)*sz)
    
        #         screen[z, x] += 1

        #     screen = screen / np.max(screen) * 255
        #     screen = screen.astype(np.uint8)
        cv2.imshow('1', screen)
        cv2.waitKey(1)



        keyPressed = plt.waitforbuttonpress(timeout=0.005)
        if keyPressed is None:
            return
        if keyPressed:
            runflag.value = 0
        else:
            if self.logger:
                self.logger.update((self.ps1, self.ps2), datatype='misc')

'''
0 1 2
  |
3-4-5
  |
6-7-8
  |
9-10-11
'''
grid = []
for i in range(0, 4):
    for j in range(0, 3):
        grid.append((j*100+50, i*100+50))
background = np.zeros((400, 300), dtype=np.uint8)

cv2.line(background, grid[3], grid[4], 50)
cv2.line(background, grid[4], grid[5], 50)
# cv2.line(background, grid[6], grid[7], 50)
# cv2.line(background, grid[7], grid[8], 50)
cv2.line(background, grid[9], grid[10], 50)
cv2.line(background, grid[10], grid[11], 50)
cv2.line(background, grid[1], grid[4], 50)
cv2.line(background, grid[4], grid[7], 50)
cv2.line(background, grid[7], grid[10], 50)

def draw_posture(code):
    # 1 for stand, 2 for bend left, 3 for bend right, 4 for sit
    
    screen = background.copy()
    if code == 1:
        cv2.line(screen, grid[1], grid[10], 255, 10)
    if code == 2:
        cv2.line(screen, grid[3], grid[4], 255, 10)
        cv2.line(screen, grid[4], grid[10], 255, 10)
    if code == 3:
        cv2.line(screen, grid[4], grid[5], 255, 10)
        cv2.line(screen, grid[4], grid[10], 255, 10)
    if code == 4:
        cv2.line(screen, grid[4], grid[10], 255, 10)
    if code == 5:
        cv2.line(screen, grid[7], grid[10], 255, 10)
    if code == 6:
        cv2.line(screen, grid[9], grid[10], 255, 10)
        cv2.line(screen, grid[10], grid[11], 255, 10)
    return screen



def compute_posture_code(ps1, ps2):
    xs1, ys1, zs1 = ps1
    xs2, ys2, zs2 = ps2
    if not isinstance(xs2, np.ndarray) or len(xs2) == 0:
        return 0
    # xs1 = np.append(xs1, xs2)
    # ys1 = np.append(ys1, ys2)
    # zs1 = np.append(zs1, zs2)
    # frame = np.asarray((xs1, ys1, zs1)).T

    xcen = stats.trim_mean(xs2, 0.1)

    # if top radar is not seeing anything
    if not isinstance(xs1, np.ndarray) or len(xs1) < 0.3*len(xs2):
        height = np.percentile(np.append(zs1, zs2), 95)
        xleft = xcen
        xright = xcen
        ratio = 0 
    else:
        height = np.percentile(zs1, 95)
        xleft = np.percentile(xs1, 2.5)
        xright = np.percentile(xs1, 97.5)
        ratio = xright - xleft / (np.percentile(xs2, 97.5)-np.percentile(xs2, 2.5))

    print(f'{height:.2f}, {ratio:.2f}')

    # 1 for stand, 2 for bend left, 3 for bend right, 4 for sit
    out = 0
    if height > 1.5:
        out = 1
    elif height > 1.2:
        to_left = xcen-xleft
        to_right = xright-xcen
        if to_left > to_right and to_left > 0.3:
            out = 2
        elif to_left < to_right and to_right > 0.3:
            out = 3
        else:
            out = 4
    elif height > 0.9:
        out = 5
    elif len(xs2) > 20:
        out = 6
    else:
        out = 0

    return out
        
