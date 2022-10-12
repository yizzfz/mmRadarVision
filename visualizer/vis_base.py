"""Base class of all visualizer modules"""
import matplotlib.pyplot as plt
import numpy as np
import traceback
import time

class Visualizer_Base():
    """Base class of a Visualizer. 
    """
    def __init__(self, queues, fm=[], xlim=[-2, 2], ylim=[0, 4], zlim=[-1, 1], 
                 radars=None,
                 logger=None, height=[], cam=None, heart_sensor=None, plotaxes=(0, 1, 2)):
        """
        Parameters:
            queues: list of data queues for the radars.
            fm: list of frame managers for the radars.
            xlim/ylim/zlim: field of view.
            radars: radar configuration that may sometimes help.
            logger: the logging module.
            height: list of height of the radars.
            cam: the camera module.
            heart_sensor: the Polar device module.
            plotaxes: which data dimensions to be visualized, default x-y-z.
        """
        self.n_radars = len(queues)
        self.queues = queues
        self.xlim = xlim
        self.ylim = ylim
        self.zlim = zlim
        self.radars = radars
        self.plotaxes = plotaxes
        if fm == []:        # should be a list of list of frame managers
            self.fm = [[] for _ in range(self.n_radars)]
        else:
            assert len(fm) == self.n_radars and "Length of Frame Managers should be equal to number of radars"
            self.fm = fm
        self.step = 0
        self.logger = logger
        self.cam = cam
        self.hs = heart_sensor
        if height == []:    # should be a list of floats
            self.height = [0 for _ in range(self.n_radars)]
        else:
            assert len(height) == self.n_radars and "Length of heights should be equal to number of radars"
            self.height = height
        # initialize the data matrix
        self.frames = [np.empty((0, 3)) for _ in range(self.n_radars)]

    def run(self, runflag):
        """Start the visualizer."""
        self.create_fig()
        self.log('start')
        self.steps = 0
        fails = 0
        try:
            while runflag.value == 1:
                update = False
                # read from the data queues of each radar
                for i, q in enumerate(self.queues):
                    # if there is a frame
                    if not q.empty():
                        update = True
                        # capture the frame
                        frame = q.get(block=True, timeout=3)
                        # apply all FMs
                        if self.fm[i]:
                            for f in self.fm[i]:
                                frame = f.run(frame)
                        # adjust the height of the point cloud based on the radar height
                        frame[:, 2] = frame[:, 2] + self.height[i]
                        self.frames[i] = frame
                        # visualize the frame of each radar
                        self.plot_each(i, frame[:, self.plotaxes], runflag)
                # if at least one radar has new data
                if update:
                    if self.logger:                 # save the data
                        self.logger.update(self.frames[:], datatype='radar')
                    if self.cam:                    # capture camera frame
                        self.cam.update(self.frames[:])
                        cam_frame = self.cam.get()
                        if self.logger:
                            self.logger.update(cam_frame, datatype='cam')
                    if self.hs and self.logger:     # capture data from Polar
                        self.logger.update(self.hs.get(), datatype='heart')
                    self.steps += 1
                    # visualize the combined frame from all radars
                    self.plot_combined(np.concatenate(self.frames, axis=0)[:, self.plotaxes], runflag)
                    fails = 0
                else:
                    fails += 1
                if fails > 10000:
                    self.log('Waiting for data')
                    time.sleep(1)
 
        except Exception as e:
            print('Exception from visualization thread:', e)
            print(traceback.format_exc())
        except KeyboardInterrupt:
            pass
        runflag.value = 0
        self.finish()

    def plot_each(self, idx, frame, runflag):
        """Any processing or plotting specifically about one radar"""
        pass
    
    def plot_combined(self, frame, runflag):
        """Any processing or plotting using fused data from all radars"""
        xs, ys, zs = np.split(frame.T, 3)
        self.fig.set_xdata(xs)
        self.fig.set_ydata(ys)
        keyPressed = plt.waitforbuttonpress(timeout=0.005)
        if keyPressed:
            runflag.value = 0

    def create_fig(self):
        """Initialize the plot window. Should match the `plot_each` and `plot_combined` functions."""
        fig = plt.figure('VB')
        ax0 = fig.add_subplot(111)
        ls, = ax0.plot([], [], '.')
        ax0.set_xlim(self.xlim)
        ax0.set_ylim(self.ylim)
        ax0.set_xlabel('x (m)')
        ax0.set_ylabel('y (m)')
        self.fig = ls

    def finish(self):
        """Called when the system exits."""
        if self.logger:
            self.logger.save()
        if self.cam:
            self.cam.stop()
        self.log('Exiting')

    def log(self, txt):
        print(f'[{self.__class__.__name__}] {txt}')