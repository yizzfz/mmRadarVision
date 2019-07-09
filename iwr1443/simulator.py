import time
import math
import matplotlib.pyplot as plt
import numpy as np

bandwidth = 4e9
T = 40e-6
S = bandwidth/T

# chirp_freq = [76e9 + i for i in range(0, bandwidth, int(bandwidth/n_step))]
# steps = [i/n_step for i in range(n_step)]
#
# chirp = [np.sin(2*np.pi*chirp_freq[i]*steps[i]+0) for i in range(n_step)]


while(1):
    dis = 50
    dis = float(input('distance to object '))
    angle = 90
    angle = float(input('angle to object '))
    wavelength = 4e-3


    dt = dis*2/3e8
    freq = 1e14*2*dis/3e8/1e6
    phase = 2*360*dis/wavelength%360

    drx = wavelength/2
    dis2 = drx*np.sin(angle/180*np.pi)
    dt2 = dt+dis2/3e8
    freq2 = 1e14*(2*dis+dis2)/3e8/1e6
    phase2 = 360*(2*dis+dis2)/wavelength%360

    print('signal back after {0:.2f} us'.format(dt*1e6))
    print('[RX1] Frequency difference {0:.2f} MHz'.format(freq))
    print('[RX1] phase difference {0:.2f} deg'.format(phase))
    print('[RX2] Frequency difference {0:.2f} MHz'.format(freq2))
    print('[RX2] phase difference {0:.2f} deg'.format(phase2))
    print()

    import pdb; pdb.set_trace()

# def d_phase(angle):
#     dis = 100
#     wavelength = 4e-3
#
#     dt = dis*2/3e8
#     freq = 1e14*2*dis/3e8/1e6
#     phase = 2*360*dis/wavelength%360
#
#     drx = wavelength/2
#     dis2 = drx*np.sin(angle/180*np.pi)
#     dt2 = dt+dis2/3e8
#     freq2 = 1e14*(2*dis+dis2)/3e8/1e6
#     phase2 = 360*(2*dis+dis2)/wavelength%360
#
#     return phase2
#
# x = np.arange(0, 90)
# y = d_phase(x)
# y2 = x*np.pi
# plt, ax = plt.subplots()
# ax.plot(x, y)
# ax.plot(x, y2)
# plt.show()
# import pdb; pdb.set_trace()
