import cv2
import matplotlib.pyplot as plt
import numpy as np
import traceback
import datetime
import pickle
from .util import *
import sys

class Visualizer_Base_2R():
    def __init__(self, queues, fm1=[], fm2=[], xlim=[-2, 2], ylim=[0, 4], zlim=[0, 2], save=False, save_start=1000):
        self.queues = queues
        self.xlim = xlim
        self.ylim = ylim
        self.zlim = zlim
        self.fm = [fm1, fm2]
        self.save = None
        self.step = 0
        self.save_start = save_start
        if save:
            timestamp = datetime.datetime.now().strftime('%m%d%H%M')
            self.save = f'./data/{timestamp}.pkl'
            self.data = []

        self.ellipses_plot1 = []
        self.ellipses_plot2 = []
        self.ellipses_vert1 = []
        self.ellipses_vert2 = []
        self.ellipses_all = []
        self.cube_drawing = []

        self.clusters1 = []
        self.clusters2 = []
        self.ff = Frame()

    def run(self, runflag):
        self.create_fig()
        self.step = 0
        while runflag.value == 1:
            try:
                update = False
                for i, q in enumerate(self.queues):
                    if not q.empty():
                        update = True
                        frame = q.get(block=True, timeout=3)
                        for f in self.fm[i]:
                            frame = f.run(frame)
                        self.plot(i, frame, runflag)
                if update:
                    self.plot_inter(runflag)
                self.step += 1


            except Exception as e:
                print('Exception from visualization thread:', e)
                print(traceback.format_exc())
                runflag.value = 0
                break

        # if self.save:
        #     with open(self.save, 'wb') as f:
        #         pickle.dump(self.data, f)

    def create_fig(self):
        fig0 = plt.figure(0)
        ax0 = fig0.add_subplot(121)
        ax1 = fig0.add_subplot(122, projection='3d')


        ls0, = ax0.plot([], [], 'r.')   # r for redn
        ls1, = ax0.plot([], [], 'bx')   # b for blue

        ax0.plot([-d_hor, 0], [0, d_ver], 'k')  # k for black
        ax0.plot([0, d_hor], [d_ver, 0], 'k')
        ax0.plot([0, d_hor], [-d_ver, 0], 'k')
        ax0.plot([0], [0], 'g+')


        ax0.set_xlim([-(d_hor+0.2), d_hor+0.2])
        ax0.set_ylim([-(d_ver+0.2), d_ver+0.2])
        ax0.set_xlabel('range (m)')
        ax0.set_ylabel('range (m)')

        radar1 = plt.Circle((0, d_ver), 0.05, color='r')
        radar2 = plt.Circle((d_hor, 0), 0.05, color='b')
        ax0.add_artist(radar1)
        ax0.add_artist(radar2)

        ax1.scatter([0], [d_ver], [radar_height], color='r', s=30)
        ax1.scatter([d_hor], [0], [radar_height], color='b', s=30)

        ax1.plot([0, 0], [d_ver, d_ver], [0, radar_height], color='r')
        ax1.plot([d_hor, d_hor], [0, 0], [0, radar_height], color='b')
        ax1.set_xlim([-(d_hor+0.2), d_hor+0.2])
        ax1.set_ylim([-(d_ver+0.2), d_ver+0.2])
        ax1.set_zlim([0, 2])
        ax1.set_xlabel('range (m)')
        ax1.set_ylabel('range (m)')
        ax1.set_zlabel('height (m)')

        self.ax0 = ax0
        self.ax1 = ax1
        self.ls0 = ls0
        self.ls1 = ls1
        plt.ion()
        plt.show()

        

    def plot(self, idx, frame, runflag):
        if idx == 0:
            xs1, ys1, zs1 = np.squeeze(np.split(frame.T, 3))
            res = rotate_and_translate(xs1, ys1, zs1, R1, T1)
            xs3, ys3, zs3 = np.squeeze(np.split(res, 3))
            
            self.ls0.set_xdata(xs3)
            self.ls0.set_ydata(ys3)
            self.clusters1 = cluster_xyz(xs3, ys3, zs3)
        elif idx == 1:
            xs2, ys2, zs2 = np.squeeze(np.split(frame.T, 3))
            res = rotate_and_translate(xs2, ys2, zs2, R2, T2)
            xs4, ys4, zs4 = np.squeeze(np.split(res, 3))

            self.ls1.set_xdata(xs4)
            self.ls1.set_ydata(ys4)
            self.clusters2 = cluster_xyz(xs4, ys4, zs4)
        else:
            print('Error: Using more than two queues, but visualizer designed for two')
            runflag.value = 0



    def plot_inter(self, runflag):
        for e in self.ellipses_all:
            e.remove()

        for c in self.cube_drawing:
            c.remove()
        self.ellipses_all = []
        cubes = []
        for c1 in self.clusters1:
            for c2 in self.clusters2:
                if c1.close_to(c2):
                    
                    cube = create_cube_from_two_clusters(c1, c2)
                    cubes.append(cube)
                    art = cube.get_bounding_box(color='green')
                    self.ellipses_all.append(art)
                    self.ax0.add_artist(art)

        self.ff.update(cubes)
        self.cube_drawing = self.ff.get_drawings()
        for c in self.cube_drawing:
            self.ax1.add_collection3d(c)


        # dispaly all boxes?
        for c in self.clusters1:
            # art = c.get_ellipse_artist(color='red')
            art = c.get_bounding_box(color='red')
            if art is not None:
                self.ellipses_all.append(art)
                self.ax0.add_artist(art)
        for c in self.clusters2:
            # art = c.get_ellipse_artist(color='blue')
            art = c.get_bounding_box(color='blue')
            if art is not None:
                self.ellipses_all.append(art)
                self.ax0.add_artist(art)


        keyPressed = plt.waitforbuttonpress(timeout=0.005)
        if keyPressed:
            runflag.value = 0
