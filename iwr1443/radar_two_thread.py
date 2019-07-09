import cv2
import serial
import time
import struct
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import multiprocessing
import queue

from sklearn.cluster import DBSCAN
from sklearn import metrics
import numpy as np
import pickle

'''
C:/ti/mmwave_sdk_01_01_00_02/packages/ti/demo/xwr16xx/mmw/docs/doxygen/html/struct_mmw_demo__output__message__header__t.html
'''


magic_word = b'\x02\x01\x04\x03\x06\x05\x08\x07'
header_size = 8 + 4 * 7

fft_freq_d = np.fft.fftfreq(240, d=1.0/7500e3)
fft_distance = fft_freq_d*3e8/(2*100e12)


def read_cfg():
    cfg = []
    with open('cfg/zoneA.cfg') as f:
        lines = f.read().split('\n')
    for line in lines:
        if not line.startswith('%'):
            cfg.append(line)
    return cfg


def send_cfg(cfg, frame_queue, message_queue, runflag):
    if os.name == 'nt':
        cfg_port_name = 'COM6'
        data_port_name = 'COM7'
    else:
        cfg_port_name = '/dev/ttyACM0'
        data_port_name = '/dev/ttyACM1'
    try:
        cfg_port = serial.Serial(cfg_port_name, 115200)
        data_port = serial.Serial(data_port_name, 921600, timeout=0.01)
    except serial.serialutil.SerialException:
        print('Failed opening serial port, check connection')
        frame_queue.put(('EXIT', None, None, None, None, None))
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

    data_to_save = []
    start = time.time()
    while runflag.value == 1:
        cnt += 1
        bytes_to_read = data_port.in_waiting
        # print(bytes_to_read)
        data_line = data_port.read(32)
        if magic_word in data_line:
            if data != b'' and send:
                out = decode_data(data, frame_queue, message_queue, ignore, 0) #enable print flag?
                data_to_save.append(out)

            data = b''
            send = 1
        data += data_line
    end = time.time()

    # with open('data.pkl', 'wb') as f:
    #     pickle.dump(data_to_save, f)
    
    # print('executed %.2f s, saved data for %.2f s' % (end-start, len(data_to_save)/5))


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
    zs = []
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
            res = parseDetectedObjects(data, tlvLength, ignore, print_flag)
            if res == None:
                return
            else:
                xs, ys, zs, doppler, ranges, peaks = res
        elif (tlvType == 2):
            parseRangeProfile(data, tlvLength)
        elif (tlvType == 6):
            parseStats(data, tlvLength)
        else:
            print("Unidentified tlv type %d" % (tlvType))
        data = data[tlvLength:]

    # if frameNum < 100:
    #     for i in range(len(xs)):
    #         ignore.append((xs[i], ys[i], zs[i]))
    # if frameNum == 100:
    #     import pdb; pdb.set_trace()
    if frame_queue.empty():
        frame_queue.put(('RUN', xs, ys, zs, peaks, ranges))

    return xs, ys, zs, peaks


def parseDetectedObjects(data, tlvLength, ignore=[], print_flag=1):
    # hreader = two unsigned short
    numDetectedObj, xyzQFormat = struct.unpack('2H', data[:4])
    xs = []
    ys = []
    zs = []
    doppler = []
    ranges = []
    peaks = []
    # print("\tDetect Obj:\t%d "%(numDetectedObj))
    for i in range(numDetectedObj):
        # print("\tObjId:\t%d "%(i))
        # each object = 6 short, 1st and 3rd being unsigned
        try:
            rangeIdx, dopplerIdx, peakVal, x, y, z = struct.unpack('HhH3h', data[4+12*i:4+12*i+12])
        except struct.error:
            print('Failed decoding data')
            return

        x = (x*1.0/(1 << xyzQFormat))
        y = (y*1.0/(1 << xyzQFormat))
        z = (z*1.0/(1 << xyzQFormat))

        condition = rangeIdx > 10 and rangeIdx < 100
        condition = (x, y, z) not in ignore
        condition = abs(z) < 0.5 and abs(x)<0.5 and y>0 and y<3 and peakVal>100 and (x, y, z) not in ignore
        condition = y > 0 and y < 3 and rangeIdx < 1025
        if condition is True:
            if print_flag:
                print("\t\tDopplerIdx:\t%d "%(dopplerIdx))
                print("\t\tRangeIdx:\t%d "%(rangeIdx))
                print("\t\tPeakVal:\t%d "%(peakVal))
                print("\t\tX (left-right):\t\t%07.3f "%(x))
                print("\t\tY (depth):\t\t%07.3f "%(y))
                print("\t\tZ (up-down):\t\t%07.3f "%(z))
                print()
                # print("%07.3f "%(y))
            xs.append(x)
            ys.append(y)
            zs.append(z)
            peaks.append(peakVal)
            doppler.append(dopplerIdx)
            ranges.append(rangeIdx)
    return xs, ys, zs, doppler, ranges, peaks


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


def vis_thread_height(frame_queue, message_queue, runflag):

    start = -1
    end = 1
    grid_h = 4
    grid_w = 7
    step = grid_w*grid_h
    step_height = (end-start)/step

    fig, axs = plt.subplots(grid_h, grid_w)

    ls = []
    plt.ion()

    h = start
    for i in range(grid_h):
        for j in range(grid_w):
            l, = axs[i, j].plot([], [], '.')
            ls.append(l)
            axs[i, j].set_xlim([-2, 2])
            axs[i, j].set_ylim([0, 4])
            title = f'{h:.3f} to {(h+step_height):.3f}'
            axs[i, j].title.set_text(title)
            h += step_height

    while runflag.value == 1:
        try:
            cmd, xs, ys, zs, peaks, ranges = frame_queue.get(block=True, timeout=3)
        except queue.Empty:
            print('Queue Empty')
            continue

        all_points = np.stack((xs, ys, zs), axis=1)

        if (cmd == 'EXIT'):
            return

        h = start
        for i in range(len(ls)):
            filtered = all_points[(all_points[:, 2] < h+step_height) & (all_points[:, 2] >= h)]
            ls[i].set_xdata(filtered[:, 0])
            ls[i].set_ydata(filtered[:, 1])
            h += step_height


        keyPressed = plt.waitforbuttonpress(timeout=0.005)
        if keyPressed:
            runflag.value = 0
            break
    plt.show()

    


def vis_thread(frame_queue, message_queue, runflag):


    fig = plt.figure()
    ax0 = fig.add_subplot(111)
    # ax1 = fig.add_subplot(212)

    plt.ion()
    ls0, = ax0.plot([], [], '.')
    # ls1, = ax1.plot([], [], '.')


    ax0.set_xlim([-2, 2])
    ax0.set_ylim([0, 4])
    ax0.set_xlabel('x (m)')
    ax0.set_ylabel('y (m)')

    # ax1.set_xlim([0, 2])
    # ax1.set_ylim([0, 5000])
    # ax1.set_xlabel('Depth (m)')
    # ax1.set_ylabel('Peak Value')

    while runflag.value == 1:
        try:
            cmd, xs, ys, zs, peaks, ranges = frame_queue.get(block=True, timeout=3)
        except queue.Empty:
            print('Queue Empty')
            break

        if (cmd == 'EXIT'):
            runflag.value = 0
            break

        # ranges1 = [np.sqrt(xs[i]**2+ys[i]**2+zs[i]**2) for i in range(len(xs))]
        # print(ranges)
        # print(ranges1)
        # print(fft_distance[ranges])


        # plt.clf()
        # plt.hist(fft_distance[ranges], 240, [0, 5])
        # plt.ylim([0, 20])
        # plt.xlim([0,])

        # print(len(xs), '============', len(ys))

        ls0.set_xdata(xs)
        ls0.set_ydata(ys)

        # ls0.set_ydata(zs)
        # ls1.set_xdata(ys)
        # ls1.set_ydata(peaks)
        keyPressed = plt.waitforbuttonpress(timeout=0.005)
        if keyPressed:
            runflag.value = 0
            break
        plt.pause(0.005)

    plt.show()


def cluster(xs, ys, zs, peaks, model):
    if (len(xs)==0):
        return [], [], [], []

    co = np.asarray([[xs[i], ys[i], zs[i]] for i in range(len(xs))])
    coo = np.asarray([[xs[i], ys[i], zs[i]] for i in range(len(xs))])

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



def data_thread(frame_queue, message_queue, runflag):
    cfg = read_cfg()
    send_cfg(cfg, frame_queue, message_queue, runflag)



def main():
    runflag = multiprocessing.Value('i', 1)
    frame_queue = multiprocessing.Queue()
    message_queue = multiprocessing.Queue()
    t0 = multiprocessing.Process(target=vis_thread_height, args=(
                                frame_queue, message_queue, runflag, ))
    t1 = multiprocessing.Process(target=data_thread, args=(
                                frame_queue, message_queue, runflag, ))
    t0.start()
    t1.start()
    t0.join()
    t1.join()


if __name__ == '__main__':
    main()
