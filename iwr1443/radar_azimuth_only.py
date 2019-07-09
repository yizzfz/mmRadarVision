import cv2
import serial
import time
import struct
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
from multiprocessing import Queue, Process
import queue

from sklearn.cluster import DBSCAN
from sklearn import metrics
import numpy as np


'''
C:/ti/mmwave_sdk_01_01_00_02/packages/ti/demo/xwr16xx/mmw/docs/doxygen/html/struct_mmw_demo__output__message__header__t.html
'''


magic_word = b'\x02\x01\x04\x03\x06\x05\x08\x07'
header_size = 8 + 4 * 7
angle_of_interest = 45.0/180.0 * np.pi

def read_cfg():
    cfg = []
    with open('cfg/azimuth.cfg') as f:
        lines = f.read().split('\n')
    for line in lines:
        if not line.startswith('%'):
            cfg.append(line)
    return cfg


def send_cfg(cfg, frame_queue, message_queue):
    try:
        cfg_port = serial.Serial('COM6', 115200)
        data_port = serial.Serial('COM7', 921600, timeout=0.01)
    except serial.serialutil.SerialException:
        print('Failed opening serial port, check connection')
        frame_queue.put(('EXIT', None, None, None, None))
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
    while 1:
        cnt += 1
        bytes_to_read = data_port.in_waiting
        # print(bytes_to_read)
        data_line = data_port.read(32)
        if magic_word in data_line:
            if data != b'' and send:
                decode_data(data, frame_queue, message_queue, ignore, 0) #enable print?
            data = b''
            send = 1
        data += data_line


def decode_data(data, frame_queue, message_queue, ignore=[], print_flag=1):
    # print('decoding')
    raw_data = data[:]

    # Q7I = one Q (unsigned long long) and seven Is (unsigned int)
    try:
        magic, version, length, platform, frameNum, cpuCycles, numObj, numTLVs = struct.unpack('Q7I', data[:header_size])
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
        try:
            tlvType, tlvLength = struct.unpack('2I', data[:8])
        except struct.error:
            print('Failed decoding TLV')
            return None
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
    try:
        numDetectedObj, xyzQFormat = struct.unpack('2H', data[:4])
    except struct.error:
        print('Failed decoding object')
        return None
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

        condition = abs(angle) < angle_of_interest and y < 3 and y > 0.2
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

def vis_thread(frame_queue, message_queue):

    fig = plt.figure()
    ax0 = fig.add_subplot(111)
    # ax1 = fig.add_subplot(212)

    plt.ion()
    ls0, = ax0.plot([], [], '.')
    # ls1, = ax1.plot([], [], '.')
    ax0.plot([0, -2], [0, np.tan(angle_of_interest)*2])
    ax0.plot([0, 2], [0, np.tan(angle_of_interest)*2])



    ax0.set_xlim([-4, 4])
    ax0.set_ylim([0, 4])
    ax0.set_xlabel('Horizontal Position (m)')
    ax0.set_ylabel('Depth (m)')

    # ax1.set_xlim([0, 2])
    # ax1.set_ylim([0, 5000])
    # ax1.set_xlabel('Depth (m)')
    # ax1.set_ylabel('Peak Value')

    # model = DBSCAN(eps=0.09, min_samples=1)

    while 1:
        try:
            cmd, xs, ys, peaks = frame_queue.get(block=True, timeout=3)
        except queue.Empty:
            print('Queue Empty')
            continue

        if (cmd == 'EXIT'):
            return

        # xs1, ys1, peaks1 = cluster(xs, ys, peaks, model)

        ls0.set_xdata(xs)
        ls0.set_ydata(ys)
        # ls1.set_xdata(ys)
        # ls1.set_ydata(peaks)
        keyPressed = plt.waitforbuttonpress(timeout=0.005)
        if keyPressed:
            break


    plt.show()


def cluster(xs, ys, peaks, model):
    if (len(xs)==0):
        return [], [], [], []

    co = np.asarray([[xs[i], ys[i]] for i in range(len(xs))])
    coo = np.asarray([[xs[i], ys[i]] for i in range(len(xs))])

    # print()
    # print(coo)

    labels = model.fit((co)).labels_
    # print(labels)
    res_x = []
    res_y = []
    res_z = []
    res_peaks = []
    for i in range(len(set(labels))):
        idx = labels==i
        res = np.average(coo[idx], axis=0)
        res_x.append(res[0])
        res_y.append(res[1])
        res_z.append(res[2])
        res_peaks.append(np.average(np.asarray(peaks)[idx]))
    # print(res_x, res_y, res_z)
    # print()
    assert(len(res_peaks)==len(res_x))

    return res_x, res_y, res_z, res_peaks



def data_thread(frame_queue, message_queue):
    cfg = read_cfg()
    send_cfg(cfg, frame_queue, message_queue)



def main():
    frame_queue = Queue()
    message_queue = Queue()
    t0 = Process(target=vis_thread, args={frame_queue, message_queue, })
    t1 = Process(target=data_thread, args={frame_queue, message_queue, })
    t0.start()
    t1.start()
    t0.join()
    t1.join()


if __name__ == '__main__':
    main()
