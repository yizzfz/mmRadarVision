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
from matplotlib.patches import Ellipse

from sklearn.cluster import DBSCAN



'''
C:/ti/mmwave_sdk_01_01_00_02/packages/ti/demo/xwr16xx/mmw/docs/doxygen/html/struct_mmw_demo__output__message__header__t.html
'''

variance_constant = 2*np.sqrt(5.991)

magic_word = b'\x02\x01\x04\x03\x06\x05\x08\x07'
header_size = 8 + 4 * 8
angle_of_interest = 45.0/180.0 * np.pi

def read_cfg():
    cfg = []
    with open('cfg/profile.cfg') as f:
        lines = f.read().split('\n')
    for line in lines:
        if not line.startswith('%'):
            cfg.append(line)
    return cfg


def send_cfg(cfg, frame_queue, runflag):
    try:
        cfg_port = serial.Serial('COM12', 115200)
        data_port = serial.Serial('COM13', 921600, timeout=0.01)
    except serial.serialutil.SerialException:
        runflag.value = 0
        print('Failed opening serial port, check connection')
        return

    assert cfg_port.is_open and data_port.is_open
    print('Hardware connected')
    for line in cfg:
        # print('[send]', line)
        line = (line+'\n').encode()
        cfg_port.write(line)
        time.sleep(0.05)

        res = ''
        while(res == '' or cfg_port.inWaiting()):
            read_len = cfg_port.inWaiting()
            res += cfg_port.read(read_len).decode()
        print(res, end='\n\n')

    data = b''
    send = 0
    print('all cfg sent')
    ignore = []
    cnt = 0
    while runflag.value==1:
        cnt += 1
        bytes_to_read = data_port.in_waiting
        # print(bytes_to_read)
        data_line = data_port.read(32)
        if magic_word in data_line:
            if data != b'' and send:
                decode_data(data, frame_queue,
                            ignore, 0)  # enable print?
            data = b''
            send = 1
        data += data_line

    cfg_port.write('sensorStop\n'.encode())



def decode_data(data, frame_queue, ignore=[], print_flag=1):
    # print('decoding')
    raw_data = data[:]

    # Q7I = one Q (unsigned long long) and seven Is (unsigned int)
    try:
        magic, version, length, platform, frameNum, cpuCycles, numObj, numTLVs, _ = struct.unpack('Q8I', data[:header_size])
    except struct.error:
        print ('Failed decoding hearder')
        return

    if print_flag:
        # os.system('clear')
        print("Packet ID:\t%d "%(frameNum))
        print("Packet len:\t%d "%(length))
    # print("TLV:\t\t%d "%(numTLVs))
    # print("Detect Obj:\t%d "%(numObj))

    if numTLVs > 1024:
        return

    data = data[header_size:]
    xs = []
    ys = []
    doppler = []
    ranges = []
    peaks = []

    for i in range(numTLVs):
        tlvType, tlvLength = struct.unpack('2I', data[:8])
        data = data[8:]
        if (tlvType == 1):
            xs, ys, doppler, ranges, peaks = parseDetectedObjects(data, tlvLength, ignore, print_flag)
        elif (tlvType == 2):
            parseRangeProfile(data, tlvLength)
        elif (tlvType == 6):
            parseStats(data, tlvLength)
        else:
            print("Unidentified tlv type %d"%(tlvType))
        data = data[tlvLength:]

    # if frameNum < 100:
    #     for i in range(len(xs)):
    #         ignore.append((xs[i], ys[i]))
    # if frameNum == 100:
    #     import pdb; pdb.set_trace()
    if frame_queue.empty():
        frame_queue.put(('RUN', xs, ys, peaks))


    return


def parseDetectedObjects(data, tlvLength, ignore=[], print_flag=1):
    # hreader = two unsigned short
    numDetectedObj, xyzQFormat = struct.unpack('2H', data[:4])
    xs = []
    ys = []
    doppler = []
    ranges = []
    peaks = []
    # print("\tDetect Obj:\t%d "%(numDetectedObj))
    for i in range(numDetectedObj):
        # print("\tObjId:\t%d "%(i))
        # each object = 6 short, 1st and 3rd being unsigned
        rangeIdx, dopplerIdx, peakVal, x, y, z = struct.unpack('HhH3h', data[4+12*i:4+12*i+12])
        x = (x*1.0/(1 << xyzQFormat))
        y = (y*1.0/(1 << xyzQFormat))
        z = (z*1.0/(1 << xyzQFormat))
        if y == 0: 
            continue
        angle = np.arctan(x/y)

        condition = abs(angle) < angle_of_interest and y < 6 and y > 0.2
        if condition is True:
            if print_flag:
                print("\t\tDopplerIdx:\t%d "%(dopplerIdx))
                print("\t\tRangeIdx:\t%d "%(rangeIdx))
                print("\t\tPeakVal:\t%d "%(peakVal))
                print("\t\tX (left-right):\t\t%07.3f "%(x))
                print("\t\tY (depth):\t\t%07.3f "%(y))
                print()
                # print("%07.3f "%(y))
            xs.append(x)
            ys.append(y)
            peaks.append(peakVal)
            doppler.append(dopplerIdx)
            ranges.append(rangeIdx)
    return xs, ys, doppler, ranges, peaks


def parseRangeProfile(data, tlvLength):
    for i in range(256):
        rangeProfile = struct.unpack('H', data[2*i:2*i+2])
        print("\tRangeProf[%d]:\t%07.3f "%(i, rangeProfile[0] * 1.0 * 6 / 8  / (1 << 8)))
    print("\tTLVType:\t%d "%(2))

def parseStats(data, tlvLength):
    interProcess, transmitOut, frameMargin, chirpMargin, activeCPULoad, interCPULoad = struct.unpack('6I', data[:24])
    print("\tOutputMsgStats:\t%d "%(6))
    print("\t\tChirpMargin:\t%d "%(chirpMargin))
    print("\t\tFrameMargin:\t%d "%(frameMargin))
    print("\t\tInterCPULoad:\t%d "%(interCPULoad))
    print("\t\tActiveCPULoad:\t%d "%(activeCPULoad))
    print("\t\tTransmitOut:\t%d "%(transmitOut))
    print("\t\tInterprocess:\t%d "%(interProcess))


def vis_thread(frame_queue, runflag):

    fig = plt.figure()
    ax0 = fig.add_subplot(111)

    plt.ion()
    ls0, = ax0.plot([], [], 'r.')
    ax0.plot([0, -6], [0, np.tan(angle_of_interest)*2])
    ax0.plot([0, 6], [0, np.tan(angle_of_interest)*2])

    model = None
    colors = []
    background_x = []
    background_y = []
    mode = 'init'
    start = time.time()



    ax0.set_xlim([-6, 6])
    ax0.set_ylim([0, 6])
    ax0.set_xlabel('Horizontal Position (m)')
    ax0.set_ylabel('Depth (m)')

    ellipses = []


    while runflag.value==1:
        try:
            cmd, xs, ys, peaks = frame_queue.get(block=True, timeout=3)
        except queue.Empty:
            print('Queue Empty')
            runflag.value = 0
            continue

        data = np.stack((xs, ys), axis=-1)

        # if mode == 'run':
        #     labels = model.predict(data)
        #     print(np.unique(labels))
        #     for i, class_idx in enumerate(np.unique(labels)):
        #         class_data = data[labels == class_idx]
        #         if class_idx == -1:
        #             ax0.plot(class_data[:, 0], class_data[:, 1], 'b.')
        #         else:
        #             ax0.plot(class_data[:, 0], class_data[:, 1], '.', color=colors[i])

        # xs1, ys1, peaks1 = cluster(xs, ys, peaks, model)

        ls0.set_xdata(xs)
        ls0.set_ydata(ys)
        # plt.pause(0.005)
        

        # if mode == 'init':
        #     background_x += xs
        #     background_y += ys
        #     if time.time() - start > 3:
        #         mode = 'run'
        #         data = np.stack(
        #             (background_x, background_y), axis=-1)

        #         centroids = cluster_DBSCAN(data)
        #         colors = get_random_colors(len(centroids))
        #         nstd = 4
        #         for (centroid, cov) in centroids:
                    
        #             vals, vecs = eigsorted(cov)
        #             theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
        #             width, height = 2 * nstd * np.sqrt(vals)
        #             ellip = Ellipse(xy=centroid, width=width,
        #                             height=height, angle=theta, alpha=0.5)
        #             ax0.add_artist(ellip)


        # data = np.stack(
        #     (xs, ys), axis=-1)

        # centroids = cluster_DBSCAN(data)
        # nstd = 4

        # for e in ellipses:
        #     e.remove()
        # ellipses = []

        # for (centroid, cov) in centroids:

        #     vals, vecs = eigsorted(cov)
        #     theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
        #     width, height = 2 * nstd * np.sqrt(vals)
        #     ellip = Ellipse(xy=centroid, width=width,
        #                     height=height, angle=theta, alpha=0.5)
        #     ax0.add_artist(ellip)
        #     ellipses.append(ellip)

                

        keyPressed = plt.waitforbuttonpress(timeout=0.005)
        if keyPressed:
            runflag.value = 0
            break

    plt.show()


def eigsorted(cov):
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    return vals[order], vecs[:, order]


def mydebug(data, model, ax):
    plt.cla()
    plt.plot([0, -6], [0, np.tan(angle_of_interest)*2])
    plt.plot([0, 6], [0, np.tan(angle_of_interest)*2])
    plt.xlim([-6, 6])
    plt.ylim([0, 6])

    label = model.labels_
    n_cluster = np.unique(label).shape[0]
    colors = get_random_colors(n_cluster)

    for i, class_idx in enumerate(np.unique(label)):
        class_data = data[label==class_idx]
        if class_idx == -1:
            plt.plot(class_data[:, 0], class_data[:, 1], 'b.')
        else:
            plt.plot(class_data[:, 0], class_data[:, 1], '.', color=colors[i])

    plt.waitforbuttonpress()


def get_random_colors(n):
    return [(np.random.rand(), np.random.rand(), np.random.rand()) for _ in range(n)]



def cluster_DBSCAN(data):
    model = DBSCAN(eps=0.2)
    model.fit((data))
    labels = model.labels_
    n_cluster = np.unique(labels).shape[0]
    centroids = []

    for i, class_idx in enumerate(np.unique(labels)):
        if class_idx != -1:
            class_data = data[labels == class_idx].T
            centroid = np.average(class_data, axis=1)
            cov = np.cov(class_data)
            centroids.append((centroid, cov))


    return centroids
    

def data_thread(frame_queue, runflag):
    cfg = read_cfg()
    send_cfg(cfg, frame_queue, runflag)



def main():
    runflag = multiprocessing.Value('i', 1)
    frame_queue = multiprocessing.Queue()
    t0 = multiprocessing.Process(target=vis_thread, args=(
                                frame_queue, runflag, ))
    t1 = multiprocessing.Process(target=data_thread, args=(
                                frame_queue, runflag, ))
    t0.start()
    t1.start()
    t0.join()
    t1.join()


if __name__ == '__main__':
    main()
