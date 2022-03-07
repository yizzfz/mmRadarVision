"""Code in this file is not guaranteed to work"""
import cv2
import matplotlib.pyplot as plt
import numpy as np
import traceback
import datetime
import pickle
from .util_visualizer import *
import sys
from .vis_cam import Visualizer_Cam_Data_Generator
from collections import Counter

AoV = 40/180*np.pi

class Visualizer_Cam_Data_Generator_TwoR(Visualizer_Cam_Data_Generator):
    """Use camera to filter out some objects in radar detection"""
    def create_fig(self):
        super().create_fig()
        self.lss = []
        for i in range(10):
            ls, = self.ax0.plot([], [])
            self.lss.append(ls)

        self.red, = self.ax0.plot([], [], '.r')
        self.blue, = self.ax0.plot([], [], '.b')

    def plot_combined(self, frame, runflag):
        _, detection = self.cam.get_detection()
        filters = None
        if detection is not None:
            filters = [((x1 - self.cam_w/2)/(self.cam_w/2) * AoV, 
                        (x2 - self.cam_w/2)/(self.cam_w/2) * AoV) 
                        for x1, y1, x2, y2 in detection
                        if y2/self.cam_h > 0.9]
            lid = 0
            for i, (x1, y1, x2, y2) in enumerate(detection):
                if y2/self.cam_h < 0.9:
                    continue
                color = self.colors[i]
                cam_left = (x1 - self.cam_w/2)/(self.cam_w/2) * AoV
                cam_right = (x2 - self.cam_w/2)/(self.cam_w/2) * AoV
                for angle in [cam_left, cam_right]:
                    if angle < 0:
                        self.lss[lid].set_xdata([0, self.xlim[1]])
                        self.lss[lid].set_ydata([d_ver, d_ver + self.xlim[1]/np.tan(angle)])
                    elif angle > 0:
                        self.lss[lid].set_xdata([0, self.xlim[0]])
                        self.lss[lid].set_ydata([
                            d_ver, d_ver + self.xlim[0]/np.tan(angle)])
                    else:
                        self.lss[lid].set_xdata([0, 0])
                        self.lss[lid].set_ydata([d_ver, self.ylim[1]])
                    lid += 1
                    # if angle < 0:
                    #     self.ax0.plot([0, self.xlim[1]], [
                    #         d_ver, d_ver + self.xlim[1]/np.tan(angle)], c=color)
                    # elif angle > 0:
                    #     self.ax0.plot([0, self.xlim[0]], [
                    #         d_ver, d_ver + self.xlim[0]/np.tan(angle)], c=color)
                    # else:
                    #     self.ax0.plot([0, 0], [d_ver, self.ylim[1]], c=color)

    

        for e in self.ellipses_all:
            e.remove()

        for (c, _, _) in self.cube_drawing:
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

        self.ff.update_with_filters(cubes, filters)
        self.cube_drawing = self.ff.get_drawings(ret_label=True)
        labels = [l for _, l, _ in self.cube_drawing]
        print('cam ', len(filters), ',', Counter(labels))

        for (c, _, _) in self.cube_drawing:
            self.ax1.add_collection3d(c)

        # dispaly all boxes?
        # for c in self.clusters1:
        #     art = c.get_bounding_box(color='red')
        #     if art is not None:
        #         self.ellipses_all.append(art)
        #         self.ax0.add_artist(art)
        # for c in self.clusters2:
        #     art = c.get_bounding_box(color='blue')
        #     if art is not None:
        #         self.ellipses_all.append(art)
        #         self.ax0.add_artist(art)

        self.save_frame()            
        keyPressed = plt.waitforbuttonpress(timeout=0.005)
        if keyPressed:
            runflag.value = 0


    def save_frame(self):
        centre_with_label = [(cen, label) for _, label, cen in self.cube_drawing]
        true_data = []
        false_data = []
        for c in self.clusters1:
            cen = c.get_centroid_xy()
            x, y = cen
            if x < -d_hor or x > d_hor or y > d_ver:
                continue
            found = False
            data = c.data.T
            data_label = 'None'
            for (cen_l, label) in centre_with_label:
                if distance_to(cen, cen_l) < 0.5:
                    found = True
                    data_label = label
                    if label == 'True':
                        true_data.append(data)
                        self.data['True'].append(data)
                        break

            if not found or data_label == 'False':
                self.data['False'].append(data)
                false_data.append(data)

        if true_data:
            true_data = np.concatenate(true_data)
            xs1, ys1, zs1 = np.squeeze(np.split(true_data.T, 3))
            self.red.set_xdata(xs1)
            self.red.set_ydata(ys1)
        else:
            self.red.set_xdata([])
            self.red.set_ydata([])


        if false_data:
            false_data = np.concatenate(false_data)
            xs2, ys2, zs2 = np.squeeze(np.split(false_data.T, 3))
            self.blue.set_xdata(xs2)
            self.blue.set_ydata(ys2)
        else:
            self.blue.set_xdata([])
            self.blue.set_ydata([])

        for key, val in self.data.items():
            print(key, len(val))


class Visualizer_Cam_Evaluation_TwoR(Visualizer_Cam_Data_Generator_TwoR):
    """Similar to last one"""
    def plot_combined(self, frame, runflag):
        _, detection = self.cam.get_detection()
        filters = None
        if detection is not None:
            filters = [((x1 - self.cam_w/2)/(self.cam_w/2) * AoV,
                        (x2 - self.cam_w/2)/(self.cam_w/2) * AoV)
                        for x1, y1, x2, y2 in detection
                        if y2/self.cam_h > 0.7]
            lid = 0
            for i, (x1, y1, x2, y2) in enumerate(detection):
                if y2/self.cam_h < 0.7:
                    continue
                color = self.colors[i]
                cam_left = (x1 - self.cam_w/2)/(self.cam_w/2) * AoV
                cam_right = (x2 - self.cam_w/2)/(self.cam_w/2) * AoV
                for angle in [cam_left, cam_right]:
                    if angle < 0:
                        self.lss[lid].set_xdata([0, self.xlim[1]])
                        self.lss[lid].set_ydata([d_ver, d_ver + self.xlim[1]/np.tan(angle)])
                    elif angle > 0:
                        self.lss[lid].set_xdata([0, self.xlim[0]])
                        self.lss[lid].set_ydata([
                            d_ver, d_ver + self.xlim[0]/np.tan(angle)])
                    else:
                        self.lss[lid].set_xdata([0, 0])
                        self.lss[lid].set_ydata([d_ver, self.ylim[1]])
                    lid += 1    

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
        drawings = self.ff.get_drawings(ret_label=True)
        self.cube_drawing = [c for (c, _, cen) in drawings if in_region(cen)]
        self.clusters1 = [c for c in self.clusters1 if in_region(c.get_centroid_xy())]

        for c in self.cube_drawing:
            self.ax1.add_collection3d(c)

        # dispaly all boxes?
        for c in self.clusters1:
            art = c.get_bounding_box(color='red')
            if art is not None:
                self.ellipses_all.append(art)
                self.ax0.add_artist(art)
        for c in self.clusters2:
            art = c.get_bounding_box(color='blue')
            if art is not None:
                self.ellipses_all.append(art)
                self.ax0.add_artist(art)

        if (len(filters) + len(self.cube_drawing) + len(self.clusters1))>0:
            print(f'cam {len(filters)}, two radar {len(self.cube_drawing)}, one radar {len(self.clusters1)}, before process {self.n_cluster_before_fm}')
            self.data.append((len(filters), len(self.cube_drawing), len(self.clusters1), self.n_cluster_before_fm))
            
        keyPressed = plt.waitforbuttonpress(timeout=0.005)
        if keyPressed:
            runflag.value = 0
