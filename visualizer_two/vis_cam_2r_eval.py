import cv2
import matplotlib.pyplot as plt
import numpy as np
import traceback
import datetime
import pickle
from .util import *
import sys
# from .vis_base_2r import Visualizer_Base_2R
from .vis_cam_2r import Visualizer_Cam_2R
from collections import Counter
import time


class Visualizer_Cam_2R_eval(Visualizer_Cam_2R):
    def __init__(self, queues, fm1=[], fm2=[], detector=None, xlim=[-(d_hor+0.2), d_hor+0.2], ylim=[-(d_ver+0.2), d_ver+0.2], zlim=[0, 2], save=False, save_start=1000):
        super().__init__(queues, fm1, fm2, detector, xlim, ylim, zlim, save, save_start)
        self.data = []
        self.n_cluster_before_fm = 0
        self.start_time = 0
        self.end_time = 0

    def run(self, runflag):
        self.create_fig()
        cv2.namedWindow("cam")
        vc = cv2.VideoCapture(0)
        if not vc.isOpened():  # try to get the first frame
            print('camera not found')
            runflag.value = 0
            return
        
        vc.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        vc.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cam_h)
        vc.set(cv2.CAP_PROP_FRAME_WIDTH, self.cam_w)
        vc.set(cv2.CAP_PROP_FPS, 30)

        w = vc.get(cv2.CAP_PROP_FRAME_WIDTH)
        h = vc.get(cv2.CAP_PROP_FRAME_HEIGHT)
        fps = vc.get(cv2.CAP_PROP_FPS)
        print('[cam] Height', h, 'Width', w, 'FPS', fps)
        self.cam_w = w
        self.cam_h = h

        self.step = 0
        warmup = 50

        while runflag.value == 1:
            try:
                rval, cam_frame = vc.read()
                detection = None
                cam_frame, detection = self.cam_process(cam_frame)
                cv2.imshow("cam", cam_frame)
                key = cv2.waitKey(1)

                update = False
                for i, q in enumerate(self.queues):
                    if not q.empty():
                        update = True
                        frame = q.get(block=True, timeout=3)
                        if i == 0 and frame is not None:
                            xs1, ys1, zs1 = np.squeeze(np.split(frame.T, 3))
                            tmp = rotate_and_translate(xs1, ys1, zs1, R1, T1)
                            xs3, ys3, zs3 = np.squeeze(np.split(tmp, 3))
                            clusters = cluster_xyz(xs3, ys3, zs3)
                            self.n_cluster_before_fm = len(clusters)
                        for f in self.fm[i]:
                            frame = f.run(frame)
                        self.plot(i, frame, runflag, plot=False)
                if update:
                    self.plot_inter(runflag, detection=detection)
                self.step += 1
                if self.step > warmup and self.start_time == 0:
                    self.start_time = time.time()

        


            except Exception as e:
                print('Exception from visualization thread:', e)
                print(traceback.format_exc())
                runflag.value = 0
                break

        self.end_time = time.time()
        e_time = self.end_time - self.start_time
        self.step -= warmup
        fps = self.step/e_time if e_time > 0 else 0
        print(f'steps {self.step}, time {e_time}, fps {fps:.2f}')

        if self.save:
            with open(self.save, 'wb') as f:
                pickle.dump(self.data, f)
                print('data saved to', self.save)

    def plot_inter(self, runflag, detection=None):
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



        if self.save:
            if (len(filters) + len(self.cube_drawing) + len(self.clusters1))>0:
                print(f'cam {len(filters)}, two radar {len(self.cube_drawing)}, one radar {len(self.clusters1)}, before process {self.n_cluster_before_fm}')
                self.data.append((len(filters), len(self.cube_drawing), len(self.clusters1), self.n_cluster_before_fm))
            
        keyPressed = plt.waitforbuttonpress(timeout=0.005)
        if keyPressed:
            runflag.value = 0
