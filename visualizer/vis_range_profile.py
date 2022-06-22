"""Plot range information at zero doppler, debugging purpose only"""
from .vis_base import Visualizer_Base
import matplotlib.pyplot as plt
import numpy as np

ADCRate = 7500e3
speedOfLight = 3e8
slope = 100e12

fft_freq_d = np.fft.fftfreq(256, d=1.0/ADCRate)
fft_distance = fft_freq_d*speedOfLight/(2*slope)

# https://e2e.ti.com/support/sensors/f/1023/p/830398/3073763
# Range profile
# Type: (MMWDEMO_OUTPUT_MSG_RANGE_PROFILE)
# Length: (Range FFT size) x(size of uint16_t)
# Value: Array of profile points at 0th Doppler(stationary objects).
# The points represent the sum of log2 magnitudes of received antennas expressed in Q9 format.

class Visualizer_Range_Profile(Visualizer_Base):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.fm:
            print('Warning: FM should not be used with range profile, ignoring')
            self.fm = None

    def plot_combined(self, frame, runflag):
        frame = frame/512
        self.ls.set_xdata(fft_distance[:128])
        self.ls.set_ydata(frame[:128])
        keyPressed = plt.waitforbuttonpress(timeout=0.005)
        if keyPressed:
            runflag.value = 0

    def create_fig(self):
        plt.ion()
        fig = plt.figure()
        ax0 = fig.add_subplot(111)

        self.ls, = ax0.plot([], [])

        ax0.set_xlim([0, 5])
        ax0.set_ylim([6, 18])
        ax0.set_xlabel('Distance (m)')
        ax0.set_ylabel('Normalised Signal Strength (dB)')
        plt.show()
