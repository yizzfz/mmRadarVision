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