import matplotlib.pyplot as plt
import numpy as np
import traceback
import time
import sys
from collections import deque
from multiprocessing import Process
from scipy.stats import trim_mean
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
import torch
sys.path.insert(1, 'C:/Users/hc13414/OneDrive - University of Bristol/mmwave/simulation/algo')
from vital_sign_processor import VitalSignProcessor
from pattern import generate_pulse
from algo import lowpass, highpass, bandpass
sys.path.insert(1, 'C:/Users/hc13414/OneDrive - University of Bristol/mmwave/heartrate/archive')
from mmhnet import MMHNet

device = torch.device('cuda')
torch.set_grad_enabled(False)

class Visualizer_Raw:
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
            print(data.shape)
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


class Visualizer_HeartRate(Visualizer_Raw):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        fps = self.config['fps']
        tracking_win = int(fps * 0.15)
        gaussian_win = int(fps * 0.7)
        downsample = 10
        self.wavelet = [('cmor1-0.5', 0.07)]
        self.VP = VitalSignProcessor(self.config, 3, tracking_win, gaussian_win, downsample=downsample)
        self.fps = int(fps)
        self.fps_ds = int(fps/downsample)
        self.win = 30
        self.Q_bpm = deque(maxlen=self.win)
        self.steps = int(self.win*self.fps)
        self.steps_ds = int(self.win*self.fps_ds)
        self.phase = np.zeros((self.steps_ds))
        self.path = np.zeros((self.fps))
        self.res = np.zeros((self.steps_ds))
        self.pulse = generate_pulse(500, 0.5)

    def make_figure(self):
        fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(16, 5), dpi=80)
        self.im = None
        ax1.set_xlim([0, self.win])
        ax1.set_ylim([-1, 1])
        self.ls1 = None
        self.ax0 = ax0
        self.ax1 = ax1
        self.ls2, = ax2.plot(np.arange(10), np.zeros(10))
        ax2.set_ylim([0, 200])
        plt.ion()
        plt.tight_layout()
        plt.show()

    def process(self, data):
        fft_mags = np.abs(data)
        fft_phases = np.angle(data)/np.pi
        phase, path = self.VP.generate_phase_signal_cont(fft_mags, fft_phases, ret_path=True)
        # res = np.zeros(phase.shape)
        # for wavelet_func, wavelet_width in self.wavelet:
        #     res = res + self.VP.wavelet(phase, border=0,
        #                     wavelet_width=wavelet_width, wavelet_func=wavelet_func)
        # res = res/res.max()
        # self.res = np.concatenate((self.res, res))[-self.steps_ds:]
        # peaks = self.VP.find_peaks(self.res, debug=False)
        # if np.mean(path) < 15 or len(peaks) < 2:
        #     bpm = 0
        # else:
        #     bpm = 60*self.fps_ds/np.average(np.diff(np.array(peaks)))
        # self.Q_bpm.append(bpm)

        self.phase = np.concatenate((self.phase, phase))[-self.steps_ds:]
        self.path = path[-self.fps:]
        self.fft_mags = data[0][-self.fps:]

    def plot(self, data):
        self.process(data)
        fft_mags = self.fft_mags
        path = self.path
        phase = self.phase
        phase = gaussian_filter1d(phase, 50)
        phase = np.convolve(phase, self.pulse, mode='same')

        # phase = phase * 100
        # phase = gaussian_filter1d(phase, 10)
        bpm_win = np.array(self.Q_bpm)
        if self.im is None:
            self.t = np.arange(0, self.win, 1/self.fps_ds)
            aspect = int(fft_mags.shape[0]/fft_mags.shape[1])
            aspect = max(1, aspect)
            self.im = self.ax0.imshow(fft_mags.T, aspect=aspect, origin='lower')
            self.ls0, = self.ax0.plot(np.arange(path.shape[0]), path, color='red')
        else:
            self.im.set_data(fft_mags.T)
            self.ls0.set_ydata(path)

        if self.ls1 is None:
            self.ls1, = self.ax1.plot(self.t, phase)
        else:
            self.ls1.set_ydata(phase)
        if bpm_win.shape[0] == 10:
            self.ls2.set_ydata(bpm_win)

        bpm = trim_mean(bpm_win, 0.1)
        if self.polar_gt:
            self.log(f'HR {bpm:.2f}, GT {self.polar_gt:.2f}')
        else:
            self.log(f'HR {bpm:.2f}')
        keyPressed = plt.waitforbuttonpress(timeout=0.005)
        if keyPressed:
            self.runflag.value = 0

class Visualizer_HeartRate_Basic(Visualizer_HeartRate):
    def process(self, data):
        fft_mags = np.abs(data)
        fft_phases = np.angle(data)/np.pi
        # peak = np.argmax(np.sum(data[0], axis=0))
        peak = 40
        path = np.ones(fft_mags.shape[0]) * peak
        phase = fft_phases[:, peak]
        phase = np.unwrap(phase, period=2)
        phase = np.concatenate(([0], np.diff(phase)))
        
        self.phase = np.concatenate((self.phase, phase))[-self.steps_ds:]
        fftfreq = np.fft.fftfreq(self.steps_ds*4, 1/self.fps_ds)
        lim0 = np.argmax(fftfreq>0.8)
        lim1 = np.argmax(fftfreq>2.5)
        self.fftfreq = fftfreq[lim0:lim1]
        self.phase_fft = np.abs(np.fft.fft(self.phase*np.hamming(self.steps_ds), self.steps_ds*4))[lim0:lim1]
        self.phase_fft = self.phase_fft/self.phase_fft.max()
        self.path = path[-self.fps:]
        self.fft_mags = fft_mags[-self.fps:]

    def make_figure(self):
        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(16, 5), dpi=80)
        self.im = None
        ax1.set_xlim([0.8, 2.5])
        ax1.set_ylim([0, 1])
        self.ls1 = None
        self.ax0 = ax0
        self.ax1 = ax1
        plt.ion()
        plt.tight_layout()
        plt.show()

    def plot(self, data):
        self.process(data)
        fft_mags = self.fft_mags
        path = self.path
        phase = self.phase
        # phase = gaussian_filter1d(phase, 5)
        # phase = phase * 10
        # phase = gaussian_filter1d(phase, 10)
        bpm_win = np.array(self.Q_bpm)
        if self.im is None:
            self.t = np.arange(0, self.win, 1/self.fps_ds)
            aspect = int(fft_mags.shape[0]/fft_mags.shape[1])
            aspect = max(1, aspect)
            self.im = self.ax0.imshow(fft_mags.T, aspect=aspect, origin='lower')
            self.ls0, = self.ax0.plot(np.arange(path.shape[0]), path, color='red')
        else:
            self.im.set_data(fft_mags.T)
            self.ls0.set_ydata(path)

        if self.ls1 is None:
            self.ls1, = self.ax1.plot(self.fftfreq, self.phase_fft)
        else:
            self.ls1.set_ydata(self.phase_fft)
        
        keyPressed = plt.waitforbuttonpress(timeout=0.005)
        if keyPressed:
            self.runflag.value = 0


class Visualizer_HeartRate_NN(Visualizer_HeartRate):
    def __init__(self, *args, **kwargs):
        self.NN = MMHNet()
        self.NN.load_state_dict(torch.load('D:/mmwave-log/hr/ckpt/31031332.pt'))
        self.NN.eval()
        self.NN.to(device)
        super().__init__(*args, **kwargs)
        self.win = 10
        self.Q_bpm = deque(maxlen=self.win)
        self.steps = int(self.win*self.fps)
        self.steps_ds = int(self.win*self.fps_ds)
        self.phase = np.zeros((self.steps_ds))
        self.path = np.zeros((self.fps))
        self.res = np.zeros((self.steps_ds))

    def process(self, data):
        fft_mags = np.abs(data)
        fft_phases = np.angle(data)/np.pi
        phase, path = self.VP.generate_phase_signal_cont(fft_mags, fft_phases, ret_path=True)
        self.phase = np.concatenate((self.phase, phase))[-self.steps_ds:]
        self.path = path[-self.fps:]
        self.fft_mags = data[0][-self.fps:]
        x = self.phase / self.phase.max()
        x = torch.tensor(x, dtype=torch.float32).reshape(1, 1, 5000).to(device)
        y = self.NN(x).cpu().numpy()[0, 0]
        if np.mean(path) < 15:
            bpm = 0
        else:
            bpm = y
        self.Q_bpm.append(bpm)

# developing version
class Visualizer_HeartRate_Alpha(Visualizer_HeartRate):
    def process(self, data):
        fft_mags = np.abs(data)
        fft_phases = np.angle(data)/np.pi
        phase, path = self.VP.generate_phase_signal_cont(fft_mags, fft_phases, ret_path=True)
        self.phase = np.concatenate((self.phase, phase))[-self.steps_ds:]
        fftfreq = np.fft.fftfreq(self.steps_ds*4, 1/self.fps_ds)
        lim0 = np.argmax(fftfreq > 0.8)
        lim1 = np.argmax(fftfreq > 2.5)
        self.fftfreq = fftfreq[lim0:lim1]
        self.phase_bp = bandpass(self.phase, cutoff=(0.8/self.fps_ds*2, 2.5/self.fps_ds*2))
        self.phase_fft = np.abs(np.fft.fft(self.phase_bp*np.hamming(self.steps_ds), self.steps_ds*4))[lim0:lim1]
        # self.phase_fft = self.phase_fft/self.phase_fft.max()
        self.path = path[-self.fps:]
        self.fft_mags = data[0][-self.fps:]

    def make_figure(self):
        fig, self.axs = plt.subplots(1, 3, figsize=(16, 5), dpi=80)
        self.axs[1].set_ylim([-0.05, 0.05])
        self.axs[2].set_ylim([0, 20])
        self.axs[2].set_xlim([0.8, 2.5])
        self.im = None
        self.ls1 = None
        self.ls2 = None
        plt.ion()
        plt.tight_layout()
        plt.show()

    def plot(self, data):
        self.process(data)
        fft_mags = self.fft_mags
        path = self.path
        phase = self.phase_bp
        # phase = gaussian_filter1d(phase, 5)
        phase = phase[::50]
        ref = 70
        if self.polar_gt:
            ref = self.polar_gt
        # phase = bandpass(phase, cutoff=((ref+10)/60/5, (ref-10)/60/5))
        # self.log(f'ref {ref}')
        # phase = phase * 10
        # phase = gaussian_filter1d(phase, 10)
        # bpm_win = np.array(self.Q_bpm)

        if self.im is None:
            # self.t = np.arange(0, self.win, 1/self.fps_ds)
            aspect = int(fft_mags.shape[0]/fft_mags.shape[1])
            aspect = max(1, aspect)
            self.im = self.axs[0].imshow(fft_mags.T, aspect=aspect, origin='lower')
            self.ls0, = self.axs[0].plot(np.arange(path.shape[0]), path, color='red')
        else:
            self.im.set_data(fft_mags.T)
            self.ls0.set_ydata(path)

        if self.ls1 is None:
            self.ls1, = self.axs[1].plot(phase)
        else:
            self.ls1.set_ydata(phase)

        if self.ls2 is None:
            self.ls2, = self.axs[2].plot(self.fftfreq, self.phase_fft)
            self.ls21, = self.axs[2].plot([ref/60, ref/60], [0, 1])
        else:
            self.ls2.set_ydata(self.phase_fft)
            self.ls21.set_data([ref/60, ref/60], [0, 20])

        # if self.ls1 is None:
        #     self.ls1, = self.ax1.plot(self.fftfreq, self.phase_fft)
        # else:
        #     self.ls1.set_ydata(self.phase_fft)

        keyPressed = plt.waitforbuttonpress(timeout=0.005)
        if keyPressed:
            self.runflag.value = 0
