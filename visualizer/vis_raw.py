import matplotlib.pyplot as plt
import numpy as np
import traceback
import time

class Visualizer_Raw:
    """For raw data display using DCA1000, e.g. for detecting heart rate"""
    def __init__(self, config, input_queue, runflag, polar=None, logger=None, dataformat='raw'):
        self.config = config
        self.input_queue = input_queue
        self.runflag = runflag
        self.polar = polar
        self.logger = logger
        self.polar_gt = None
        self.dataformat = dataformat
        if self.logger:
            self.logger.set_header({'config': self.config})

    def run(self):
        self.make_figure()
        try:
            while self.runflag.value == 1:
                if not self.input_queue.empty():
                    if self.input_queue.qsize() > 10:
                        self.log(f'Warning: input queue size {self.input_queue.qsize()}')
                    data = self.input_queue.get(block=True)     # data shape (2, n_chirps, n_samples), 2 for fft mags and phases
                    if self.polar:
                        self.polar_gt = self.polar.get()
                    if self.logger:
                        self.logger.update(data, datatype='radar-raw')
                        if self.polar:
                            self.logger.update(self.polar_gt, datatype='polar')
                    self.plot(data)
                else:
                    time.sleep(0.05)
        except Exception as e:
            print('Exception from visualization thread:', e)
            print(traceback.format_exc())
        except KeyboardInterrupt:
            pass
        self.runflag.value = 0
        self.exit()

    def exit(self):
        if self.logger:
            self.logger.save()
        # time.sleep(2)
        # while not self.input_queue.empty():
        #     self.input_queue.get()
        self.log('exited')
            
    def make_figure(self):
        self.fig = plt.figure('Raw data')
        self.ax = self.fig.add_subplot(111)
        self.im = None
        plt.ion()
        plt.show()

    def plot(self, data):
        data = data[0]      # only plot the first rx
        if self.dataformat == 'fft':
            self.plot_fft(data)
        elif self.dataformat == 'raw':
            self.plot_raw(data)

        if self.polar_gt:
            self.log(f'GT {self.polar_gt:.2f}')
        keyPressed = plt.waitforbuttonpress(timeout=0.005)
        if keyPressed:
            self.runflag.value = 0

    def plot_fft(self, data):
        data = np.abs(data)      # take the fft mags
        if self.im is None:
            aspect = int(data.shape[0]/data.shape[1])
            aspect = max(aspect, 1)
            self.im = self.ax.imshow(data.T, aspect=aspect, origin='lower')
        else:
            self.im.set_data(data.T)
            
    def plot_raw(self, data):
        chirp0 = np.abs(data[0])
        if self.im is None:
            self.im,  = plt.plot(chirp0)
        else:
            self.im.set_ydata(chirp0)


    def log(self, txt):
        print(f'[{self.__class__.__name__}] {txt}')

class Visualizer_Raw_Pointcloud(Visualizer_Raw):
    """For raw data display using DCA1000 + point cloud"""
    def __init__(self, config, dca1000_queue, radar_queue, runflag, polar=None, logger=None, dataformat='raw'):
        self.config = config
        self.dca1000_queue = dca1000_queue
        self.radar_queue = radar_queue
        self.runflag = runflag
        self.polar = polar
        self.logger = logger
        self.polar_gt = None
        self.dataformat = dataformat
        if self.logger:
            self.logger.set_header({'config': self.config})

    def run(self):
        self.make_figure()
        try:
            while self.runflag.value == 1:
                if not self.dca1000_queue.empty():
                    if self.dca1000_queue.qsize() > 10:
                        self.log(f'Warning: dca1000 queue size {self.dca1000_queue.qsize()}')
                    data = self.dca1000_queue.get(block=True)     # data shape (2, n_chirps, n_samples), 2 for fft mags and phases
                    if self.polar:
                        self.polar_gt = self.polar.get()
                    if self.logger:
                        self.logger.update(data, datatype='radar-raw')
                        if self.polar:
                            self.logger.update(self.polar_gt, datatype='polar')
                    self.plot_raw(data)
                elif not self.radar_queue.empty():
                    if self.radar_queue.qsize() > 10:
                        self.log(f'Warning: radar queue size {self.radar_queue.qsize()}')
                    data = self.radar_queue.get(block=True)     # pointcloud, timestamp
                    if self.logger:
                        self.logger.update(data, datatype='radar-pc')
                    self.plot_pc(data)
                else:
                    time.sleep(0.05)
        except Exception as e:
            print('Exception from visualization thread:', e)
            print(traceback.format_exc())
        except KeyboardInterrupt:
            pass
        self.runflag.value = 0
        self.exit()
            
    def make_figure(self):
        fig, axs = plt.subplots(1, 2)
        self.ax0 = axs[0]
        self.ax1 = axs[1]
        self.im0 = None
        self.ax1.set_xlim([-2, 2])
        self.ax1.set_ylim([-2, 2])
        self.ax1.set_xlabel('x (m)')
        self.ax1.set_ylabel('height (m)')
        self.ls1, = self.ax1.plot([], [], 'r.')   # r for red
        plt.ion()
        plt.show()

    def plot_pc(self, frame):
        frame = frame[0][:, :3]
        if frame.shape[0] > 1:
            xs1, ys1, zs1 = np.squeeze(np.split(frame, 3, axis=-1))
            self.ls1.set_xdata(xs1)
            self.ls1.set_ydata(zs1)
        keyPressed = plt.waitforbuttonpress(timeout=0.005)
        if keyPressed:
            self.runflag.value = 0

    def plot_raw(self, data):
        chirp0 = np.abs(data[0, 0])
        if self.im0 is None:
            self.im0, = self.ax0.plot(chirp0)
        else:
            self.im0.set_ydata(chirp0)
        keyPressed = plt.waitforbuttonpress(timeout=0.005)
        if keyPressed:
            self.runflag.value = 0
