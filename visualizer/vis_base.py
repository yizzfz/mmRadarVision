import cv2
import matplotlib.pyplot as plt
import numpy as np
import traceback
import datetime
import pickle

class Visualizer_Base():
    def __init__(self, queues, fm=None, xlim=[-2, 2], ylim=[0, 4], zlim=[0, 2], save=False, save_start=1000):
        self.queues = queues
        self.xlim = xlim
        self.ylim = ylim
        self.zlim = zlim
        self.fm = fm
        self.save = None
        self.step = 0
        self.save_start = save_start
        if save:
            timestamp = datetime.datetime.now().strftime('%m%d%H%M')
            self.save = f'./data/{timestamp}.pkl'

    def run(self, runflag):
        fig = self.create_fig()
        data_to_save = []
        self.step = 0
        while runflag.value == 1:
            try:
                data = []
                for i, q in enumerate(self.queues):
                    if not q.empty():
                        frame = q.get(block=True, timeout=3)
                        data.append(frame)
                        if self.fm:
                            for f in self.fm:
                                frame = f.run(frame)
                                data.append(frame)
                        self.plot(i, fig, frame, runflag)
                if self.save and self.step >= self.save_start:
                    data_to_save.append(data)
                self.step += 1


            except Exception as e:
                print('Exception from visualization thread:', e)
                print(traceback.format_exc())
                runflag.value = 0
                break

        if self.save:
            with open(self.save, 'wb') as f:
                pickle.dump(data_to_save, f)

    def plot(self, idx, fig, frame):
        return

    def create_fig(self):
        return plt.figure()
        

class Visualizer_Multi_Base():
    def __init__(self, queues, fm=None, xlim=[-2, 2], ylim=[0, 4], zlim=[0, 2], n_row=1, n_col=2):
        self.queues = queues
        self.xlim = xlim
        self.ylim = ylim
        self.zlim = zlim
        self.fm = fm
        self.n_row = n_row
        self.n_col = n_col
        self.n = n_row * n_col

    def run(self, runflag):
        fig = self.create_fig()
        while runflag.value == 1:
            try:
                for i, q in enumerate(self.queues):
                    if not q.empty():
                        frame = q.get(block=True, timeout=3)
                        if self.fm:
                            frames_to_draw = []
                            for f in self.fm:
                                frame = f.run(frame)
                                frames_to_draw.append(frame)
                        self.plot(i, fig, frames_to_draw, runflag)

            except Exception as e:
                print('Exception from visualization thread:', e)
                print(traceback.format_exc())
                runflag.value = 0
                break

    def plot(self, idx, fig, frame):
        return

    def create_fig(self):
        return plt.figure()


class Visualizer_3D_Base(Visualizer_Base):
    def run(self, runflag):
        fig = self.create_fig()
        while runflag.value == 1:
            try:
                for i, q in enumerate(self.queues):
                    if not q.empty():
                        frame = q.get(block=True, timeout=3)
                        if self.fm:
                            for f in self.fm:
                                frame = f.run(frame)
                        self.plot(i, fig, frame, runflag)

            except Exception as e:
                print('Exception from visualization thread:', e)
                print(traceback.format_exc())
                runflag.value = 0
                break


class Visualizer_Cam_Base():
    def __init__(self, queues, fm=None, xlim=[-2, 2], ylim=[0, 4], zlim=[0, 2], 
                 detector=None, detector_start=1000, save=False, save_start=1000):
        self.queues = queues
        self.xlim = xlim
        self.ylim = ylim
        self.zlim = zlim
        self.fm = fm
        self.detector = detector
        self.detector_start = detector_start
        self.step = 0
        self.save = None
        self.save_start = save_start
        if save:
            timestamp = datetime.datetime.now().strftime('%m%d%H%M')
            self.save = f'./data/{timestamp}.pkl'

    def run(self, runflag):
        fig = self.create_fig()
        cv2.namedWindow("cam")
        vc = cv2.VideoCapture(0)
        if not vc.isOpened():  # try to get the first frame
            print('camera not found')
            runflag.value = 0
            return
        
        w = vc.get(cv2.CAP_PROP_FRAME_WIDTH)
        h = vc.get(cv2.CAP_PROP_FRAME_HEIGHT)
        fps = vc.get(cv2.CAP_PROP_FPS)
        print('[cam] Height', h, 'Width', w, 'FPS', fps)
        self.cam_w = w
        self.cam_h = h
        data_to_save = []
        self.step = 0
        while runflag.value == 1:
            try:
                data = []
                rval, cam_frame = vc.read()
                res = None
                if self.step >= self.detector_start:
                    cam_frame, res = self.cam_process(cam_frame)
                cv2.imshow("cam", cv2.cvtColor(cam_frame, cv2.COLOR_BGR2GRAY))
                key = cv2.waitKey(1)
                data.append(cv2.cvtColor(cam_frame, cv2.COLOR_BGR2GRAY))

                for i, q in enumerate(self.queues):
                    if not q.empty():
                        frame = q.get(block=True, timeout=3)
                        data.append(frame)
                        if self.fm:
                            for f in self.fm:
                                frame = f.run(frame)
                                data.append(frame)
                        self.plot(i, fig, frame, runflag, detection=res)

                            
                if key == 27:  # exit on ESC
                    runflag.value = 0
                    break
                self.step += 1
                if self.save and self.step >= self.save_start:
                    data_to_save.append(data)


            except Exception as e:
                print('Exception from visualization thread:', e)
                print(traceback.format_exc())
                runflag.value = 0
                break

        
        if self.save:
            with open(self.save, 'wb') as f:
                pickle.dump(data_to_save, f)
        cv2.destroyWindow("preview")
        vc.release()

    def cam_process(self, frame):
        if self.detector is None:
            return frame, None
        else:
            return self.detector.process(frame)

    def plot(self, idx, fig, frame, cam=None):
        return

    def create_fig(self):
        return plt.figure()
