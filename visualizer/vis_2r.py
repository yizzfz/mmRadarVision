"""Using two radars. Used in paper https://doi.org/10.1109/MAES.2020.3021322"""
import matplotlib.pyplot as plt
import numpy as np
import traceback
import datetime
import pickle
from .util_visualizer import *
from .vis_base import Visualizer_Base
from scipy.spatial.transform import Rotation as R


class Visualizer_TwoR(Visualizer_Base):
    """Display points from two radars"""
    def __init__(self, *args, plot_mode='full', show3dpoints=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.plot_mode = plot_mode
        self.show3dpoints = show3dpoints
        self.ellipses_plot1 = []
        self.ellipses_plot2 = []
        self.ellipses_vert1 = []
        self.ellipses_vert2 = []
        self.ellipses_all = []
        self.cube_drawing = []
        self.clusters1 = []
        self.clusters2 = []
        self.ff = Frame()

    def create_fig(self):
        # create a 2D x-y figure and a 3D figure
        fig0 = plt.figure(0)
        ax0 = fig0.add_subplot(121)
        ax1 = fig0.add_subplot(122, projection='3d')

        ls0, = ax0.plot([], [], 'r.')   # r for redn
        ls1, = ax0.plot([], [], 'bx')   # b for blue

        # plot the room layout as in the paper
        ax0.plot([-d_hor, 0], [0, d_ver], 'k')  # k for black
        ax0.plot([0, d_hor], [d_ver, 0], 'k')
        ax0.plot([0, d_hor], [-d_ver, 0], 'k')
        ax0.plot([0], [0], 'g+')

        ax0.set_xlim(self.xlim)
        ax0.set_ylim(self.ylim)
        ax0.set_xlabel('range (m)')
        ax0.set_ylabel('range (m)')

        # plot the location of the two radars
        radar1 = plt.Circle((0, d_ver), 0.05, color='r')
        radar2 = plt.Circle((d_hor, 0), 0.05, color='b')
        ax0.add_artist(radar1)
        ax0.add_artist(radar2)

        # plot the room layout and radar location in 3D
        ax1.scatter([0], [d_ver], [self.height[0]], color='r', s=30)
        ax1.scatter([d_hor], [0], [self.height[1]], color='b', s=30)
        ax1.plot([0, 0], [d_ver, d_ver], [0, self.height[0]], color='r')
        ax1.plot([d_hor, d_hor], [0, 0], [0, self.height[1]], color='b')
        ax1.set_xlim(self.xlim)
        ax1.set_ylim(self.ylim)
        ax1.set_zlim(self.zlim)
        ax1.set_xlabel('range (m)')
        ax1.set_ylabel('range (m)')
        ax1.set_zlabel('height (m)')

        self.ax0 = ax0
        self.ax1 = ax1
        self.ls0 = ls0
        self.ls1 = ls1
        
        plt.ion()
        plt.show()
    

    def plot_each(self, idx, frame, runflag):
        if frame is None:
            return
        if idx == 0:    # radar 1
            xs1, ys1, zs1 = np.squeeze(np.split(frame.T, 3))
            # transform the points to have the same coordinate system
            # res = rotate_and_translate(xs1, ys1, zs1, R1, T1)
            # xs3, ys3, zs3 = np.squeeze(np.split(res, 3))
            if self.plot_mode == 'full':    # plot all points
                self.ls0.set_xdata(xs1)
                self.ls0.set_ydata(ys1)
            # DBSCAN clsutering the points
            self.clusters1 = cluster_xyz(xs1, ys1, zs1)
        elif idx == 1:  # radar 2
            xs2, ys2, zs2 = np.squeeze(np.split(frame.T, 3))
            # res = rotate_and_translate(xs2, ys2, zs2, R2, T2)
            # xs4, ys4, zs4 = np.squeeze(np.split(res, 3))
            if self.plot_mode == 'full':
                self.ls1.set_xdata(xs2)
                self.ls1.set_ydata(ys2)
            self.clusters2 = cluster_xyz(xs2, ys2, zs2)
        else:
            print('Error: Using more than two queues, but visualizer designed for two')
            runflag.value = 0

    def plot_combined(self, frame, runflag):
        show3dpoints = self.show3dpoints
        if show3dpoints:        # plot all points in 3D
            self.ax1.cla()
            self.ax1.scatter([0], [d_ver], [self.height[0]], color='r', s=30)
            self.ax1.scatter([d_hor], [0], [self.height[1]], color='b', s=30)

            self.ax1.plot([0, 0], [d_ver, d_ver], [0, self.height[0]], color='r')
            self.ax1.plot([d_hor, d_hor], [0, 0], [0, self.height[1]], color='b')
            self.ax1.set_xlim(self.xlim)
            self.ax1.set_ylim(self.ylim)
            self.ax1.set_zlim(self.zlim)
            self.ax1.set_xlabel('range (m)')
            self.ax1.set_ylabel('range (m)')
            self.ax1.set_zlabel('height (m)')

        for e in self.ellipses_all:     # remove old ellipses
            e.remove()

        for c in self.cube_drawing:     # remove old cubes
            c.remove()
        self.ellipses_all = []
        cubes = []
        # loop through all pairs of cubes from two radars
        for c1 in self.clusters1:
            for c2 in self.clusters2:
                if c1.close_to(c2):     # if the two cubes are close, that is a valid detection
                    cube = create_cube_from_two_clusters(c1, c2)
                    cubes.append(cube)
                    if self.plot_mode == 'full':    # plot rectangle
                        art = cube.get_bounding_box(color='green')
                    if self.plot_mode == 'simple':  # plot centroid only
                        cen = cube.get_centroid_xy()
                        art = plt.Circle(cen, 0.1, color='green')
                    self.ellipses_all.append(art)
                    self.ax0.add_artist(art)

        self.ff.update(cubes)           # update the scene
        self.cube_drawing = self.ff.get_drawings()
        for c in self.cube_drawing:     # draw the detection
            self.ax1.add_collection3d(c)
        objs = self.ff.get_objs()
        cens = [obj.get_centroid_xy() for obj in objs]
        if self.logger:
            self.logger.update(cens, datatype='misc')

        # dispaly all boxes?
        # for c in self.clusters1:
        #     # art = c.get_ellipse_artist(color='red')
        #     art = c.get_bounding_box(color='red')
        #     if art is not None:
        #         self.ellipses_all.append(art)
        #         self.ax0.add_artist(art)
        # for c in self.clusters2:
        #     # art = c.get_ellipse_artist(color='blue')
        #     art = c.get_bounding_box(color='blue')
        #     if art is not None:
        #         self.ellipses_all.append(art)
        #         self.ax0.add_artist(art)

        if show3dpoints and self.frames[0] is not None and self.frames[1] is not None:
            xs, ys, zs = np.split(self.frames[0].T, 3)
            self.ax1.scatter(xs, ys, zs, color='red')
            xs, ys, zs = np.split(self.frames[1].T, 3)
            self.ax1.scatter(xs, ys, zs, color='blue')

        keyPressed = plt.waitforbuttonpress(timeout=0.005)
        if keyPressed:
            runflag.value = 0


class Visualizer_TwoR_Tracker(Visualizer_TwoR):
    """A human tracker using two radars, x-y plot"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def create_fig(self):
        # same as Visualizer_TwoR, only 2D plot
        fig0 = plt.figure(0)
        ax0 = fig0.add_subplot(111)

        ls0, = ax0.plot([], [], 'r.')   # r for redn
        ls1, = ax0.plot([], [], 'bx')   # b for blue

        ax0.plot([-d_hor, 0], [0, d_ver], 'k')  # k for black
        ax0.plot([0, d_hor], [d_ver, 0], 'k')
        ax0.plot([0, d_hor], [-d_ver, 0], 'k')
        ax0.plot([0], [0], 'g+')

        ax0.set_xlim(self.xlim)
        ax0.set_ylim(self.ylim)
        ax0.set_xlabel('range (m)')
        ax0.set_ylabel('range (m)')

        radar1 = plt.Circle((0, d_ver), 0.05, color='r')
        radar2 = plt.Circle((d_hor, 0), 0.05, color='b')
        ax0.add_artist(radar1)
        ax0.add_artist(radar2)

        self.ax0 = ax0
        self.ls0 = ls0
        self.ls1 = ls1

        plt.ion()
        plt.show()

    def plot_combined(self, frame, runflag):
        # plot the trace of the detected object
        cubes = []
        for c1 in self.clusters1:
            for c2 in self.clusters2:
                if c1.close_to(c2):
                    cube = create_cube_from_two_clusters(c1, c2)
                    cubes.append(cube)

        self.ff.update(cubes)
        objs = self.ff.get_objs()

        if len(objs) > 0:
            history = objs[0].get_history()
            history = np.asarray(history)
            self.ls0.set_xdata(history[:, 0])
            self.ls0.set_ydata(history[:, 1])

        if len(objs) > 1:
            history = objs[1].get_history()
            history = np.asarray(history)
            self.ls1.set_xdata(history[:, 0])
            self.ls1.set_ydata(history[:, 1])
        
        keyPressed = plt.waitforbuttonpress(timeout=0.005)
        if keyPressed:
            runflag.value = 0

class Visualizer_TwoR_2D(Visualizer_TwoR):
    def create_fig(self):
        # create a 2D x-y figure and a 3D figure
        fig0 = plt.figure(0)
        ax0 = fig0.add_subplot(111)

        ls0, = ax0.plot([], [], 'r.')   # r for red
        ls1, = ax0.plot([], [], 'bx')   # b for blue

        # plot the radar layout
        radar_location = [r[-1] for r in self.radars]
        radar_direction = [r[-2] for r in self.radars]
        r1, r2 = radar_location
        rm1 = R.from_euler('xyz', radar_direction[0], degrees=True).as_matrix()
        rm2 = R.from_euler('xyz', radar_direction[1], degrees=True).as_matrix()
        ax0.plot([0], [0], 'g+')

        ax0.set_xlim(self.xlim)
        ax0.set_ylim(self.ylim)
        ax0.set_xlabel('range (m)')
        ax0.set_ylabel('range (m)')

        # plot the location of the two radars
        radar1 = plt.Circle((r1[:2]), 0.05, color='r', label='Radar 1')
        radar2 = plt.Circle((r2[:2]), 0.05, color='b', label='Radar 2')
        ax0.add_artist(radar1)
        ax0.add_artist(radar2)
        ax0.legend(loc='upper left')

        # 60 degree fov
        left = np.asarray([[-1, 0.58, 0]])*0.3
        right = np.asarray([[1, 0.58, 0]])*0.3
        left1 = (left @ rm1 + r1)[0]
        right1 = (right @ rm1 + r1)[0]
        left2 = (left @ rm2 + r2)[0]
        right2 = (right @ rm2 + r2)[0]
        print(left, right)
        print(r1, left1, right1, radar_direction[0])
        print(r2, left2, right2, radar_direction[1])
        plt.plot([r1[0], left1[0]], [r1[1], left1[1]], 'k--')
        plt.plot([r1[0], right1[0]], [r1[1], right1[1]], 'k--')
        plt.plot([r2[0], left2[0]], [r2[1], left2[1]], 'k--')
        plt.plot([r2[0], right2[0]], [r2[1], right2[1]], 'k--')

        self.ax0 = ax0
        self.ls0 = ls0
        self.ls1 = ls1

        plt.ion()
        plt.show()

    def plot_combined(self, frame, runflag):
        for e in self.ellipses_all:     # remove old ellipses
            e.remove()
        self.ellipses_all = []
        cubes = []
        # loop through all pairs of cubes from two radars
        for c1 in self.clusters1:
            for c2 in self.clusters2:
                if c1.close_to(c2):     # if the two cubes are close, that is a valid detection
                    cube = create_cube_from_two_clusters(c1, c2)
                    cubes.append(cube)
                    if self.plot_mode == 'full':    # plot rectangle
                        art = cube.get_bounding_box(color='green')
                    if self.plot_mode == 'simple':  # plot centroid only
                        cen = cube.get_centroid_xy()
                        art = plt.Circle(cen, 0.1, color='green')
                    self.ellipses_all.append(art)
                    self.ax0.add_artist(art)

        self.ff.update(cubes)           # update the scene
        objs = self.ff.get_objs()
        cens = [obj.get_centroid_xy() for obj in objs]
        if self.logger:
            self.logger.update(cens, datatype='misc')

        # dispaly all boxes?
        # for c in self.clusters1:
        #     # art = c.get_ellipse_artist(color='red')
        #     art = c.get_bounding_box(color='red')
        #     if art is not None:
        #         self.ellipses_all.append(art)
        #         self.ax0.add_artist(art)
        # for c in self.clusters2:
        #     # art = c.get_ellipse_artist(color='blue')
        #     art = c.get_bounding_box(color='blue')
        #     if art is not None:
        #         self.ellipses_all.append(art)
        #         self.ax0.add_artist(art)

        keyPressed = plt.waitforbuttonpress(timeout=0.005)
        if keyPressed:
            runflag.value = 0
