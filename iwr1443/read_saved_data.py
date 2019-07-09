import cv2
import serial
import time
import struct
import os
import multiprocessing
import queue

import numpy as np
import pickle
import platform
import pdb
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN



filename = '5min-1m.pkl'
fft_freq_d = np.fft.fftfreq(256, d=1.0/7500e3)
fft_distance = fft_freq_d*3e8/(2*100e12)

# xs, ys, zs, dopplers, ranges, peaks

def main():
    with open(filename, 'rb') as f:
        data = pickle.load(f)

    x_all = []
    y_all = []
    z_all = []
    ranges_all = []
    for frame in data:
        # pdb.set_trace()
        ranges = frame[4]
        # plt.clf()
        # plt.hist(fft_distance[ranges], 256, [0, 5])
        # plt.ylim([0, 20])
        # plt.pause(0.005)
        ranges_all += (ranges)
        x_all += frame[0]
        y_all += frame[1]
        z_all += frame[2]


    plt.hist(fft_distance[ranges_all], 100, [0, 2])
    plt.show()

    # FoV_h = 28
    # FoV_v = 15
    # FoVrh = FoV_h/180*np.pi
    # FoVrv = FoV_v/180*np.pi
    # scene1 = cv2.imread('5min-lab1.png')
    # scene2 = scene1.copy()

    # w = scene1.shape[1]
    # h = scene1.shape[0]
    # data = np.stack((x_all, y_all, z_all), axis=-1)
    # model = cluster(data)
    
    # label = model.labels_
    # n_cluster = np.unique(label).shape[0]
    # print(np.unique(label), n_cluster)


    # for i in range(data.shape[0]):
    #     x, y, z = data[i]
    #     angle_h = np.arctan(x/y)
    #     angle_v = np.arctan(z/y)
    #     x = int((angle_h/FoVrh + 1) * 0.5*w)
    #     z = int((-angle_v/FoVrv + 1) * 0.5*h)
    #     cv2.circle(scene1, (int(x), int(z)), 5, (255, 255, 255), -1)

    #     class_idx = label[i]
    #     color = int(255*class_idx/n_cluster) if class_idx != -1 else 255
    #     cv2.circle(scene2, (int(x), int(z)), 5, (color, color, color), -1)



    # cv2.imshow('win1', scene1)
    # cv2.imshow('win2', scene2)
    # cv2.waitKey()


    
def cluster(data):
    model = DBSCAN(eps=0.1)
    model.fit((data))
    return model








if __name__ == '__main__':
    main()
