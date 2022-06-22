"""Base class of all visualizer modules"""
import matplotlib.pyplot as plt
import numpy as np
import traceback
import time

class Visualizer_Base():
    """Base class. 
    """
    def __init__(self, queues, fm=[], xlim=[-2, 2], ylim=[0, 4], zlim=[-1, 1], 
                 logger=None, height=[], cam=None, heart_sensor=None):
        self.n_radars = len(queues)
        self.queues = queues
        self.xlim = xlim
        self.ylim = ylim
        self.zlim = zlim
        if fm == []:
            self.fm = [[] for _ in range(self.n_radars)]
        else:
            assert len(fm) == self.n_radars and "Length of Frame Managers should be equal to number of radars"
            self.fm = fm
        self.step = 0
        self.logger = logger
        self.cam = cam
        self.hs = heart_sensor
        if height == []:
            self.height = [0 for _ in range(self.n_radars)]
        else:
            assert len(height) == self.n_radars and "Length of heights should be equal to number of radars"
            self.height = height
        self.frames = [np.empty((0, 3)) for _ in range(self.n_radars)]

    def run(self, runflag):
        self.create_fig()
        self.log('start')
        self.steps = 0
        fails = 0
        while runflag.value == 1:
            try:
                update = False
                for i, q in enumerate(self.queues):
                    if not q.empty():
                        update = True
                        frame = q.get(block=True, timeout=3)
                        if self.fm[i]:
                            for f in self.fm[i]:
                                frame = f.run(frame)

                        frame[:, 2] = frame[:, 2] + self.height[i]
                        self.frames[i] = frame
                        self.plot_each(i, frame, runflag)
                if update:
                    if self.logger:
                        self.logger.update(self.frames, datatype='radar')
                    if self.cam:
                        self.cam.update(self.frames)
                        cam_frame = self.cam.get()
                        self.logger.update(cam_frame, datatype='cam')
                    if self.hs:
                        self.logger.update(self.hs.get(), datatype='heart')
                    self.steps += 1
                    self.plot_combined(np.concatenate(self.frames, axis=0), runflag)
                    fails = 0
                else:
                    fails += 1
                if fails > 10000:
                    self.log('Waiting for data')
                    time.sleep(1)
 
            except Exception as e:
                print('Exception from visualization thread:', e)
                print(traceback.format_exc())
                runflag.value = 0
                break
        self.finish()

    def plot_each(self, idx, frame, runflag):
        """any processing or plotting specifically about one radar"""
        pass
    
    def plot_combined(self, frame, runflag):
        """any processing or plooting using fused data from all radars"""
        xs, ys, zs = np.split(frame.T, 3)
        self.fig.set_xdata(xs)
        self.fig.set_ydata(ys)
        keyPressed = plt.waitforbuttonpress(timeout=0.005)
        if keyPressed:
            runflag.value = 0

    def create_fig(self):
        fig = plt.figure('VB')
        ax0 = fig.add_subplot(111)
        ls, = ax0.plot([], [], '.')
        ax0.set_xlim(self.xlim)
        ax0.set_ylim(self.ylim)
        ax0.set_xlabel('x (m)')
        ax0.set_ylabel('y (m)')
        self.fig = ls

    def finish(self):
        if self.logger:
            self.logger.save()

    def log(self, txt):
        print(f'[{self.__class__.__name__}] {txt}')