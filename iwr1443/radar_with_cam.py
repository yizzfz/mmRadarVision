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

'''
C:/ti/mmwave_sdk_01_01_00_02/packages/ti/demo/xwr16xx/mmw/docs/doxygen/html/struct_mmw_demo__output__message__header__t.html
'''


magic_word = b'\x02\x01\x04\x03\x06\x05\x08\x07'
header_size = 8 + 4 * 7
fps = 5
device = platform.uname()
fft_freq_d = np.fft.rfftfreq(1024, d=1.0/7500e3)*2
fft_distance = fft_freq_d*3e8/(2*100e12)

save_data = False




def read_cfg():
    cfg = []
    # with open('cfg/best_range_res.cfg') as f:
    with open('cfg/zone.cfg') as f:

        lines = f.read().split('\n')
    for line in lines:
        if not line.startswith('%'):
            cfg.append(line)
    return cfg


def send_cfg(cfg, frame_queue, message_queue, runflag):
    if device[0] == 'Windows':
        if device[1] == 'IT070107':
            cfg_port_name = 'COM5'
            data_port_name = 'COM4'
        else:
            cfg_port_name = 'COM6'
            data_port_name = 'COM7'
    else:
        cfg_port_name = '/dev/ttyACM0'
        data_port_name = '/dev/ttyACM1'
    print('connecting to', cfg_port_name, data_port_name)
    try:
        cfg_port = serial.Serial(cfg_port_name, 115200)
        data_port = serial.Serial(data_port_name, 921600, timeout=0.01)
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

    data_to_save = []
    start = time.time()
    while runflag.value == 1:
        cnt += 1
        bytes_to_read = data_port.in_waiting
        # print(bytes_to_read)
        data_line = data_port.read(32)
        if magic_word in data_line:
            if data != b'' and send:
                # enable print flag?
                out = decode_data(data, frame_queue, message_queue, ignore, 0)
                if time.time() - start > 30 and time.time() - start < 330:
                    data_to_save.append(out)

            data = b''
            send = 1
        data += data_line

    cfg_port.write('sensorStop\n'.encode())
    end = time.time()

    if save_data:
        with open('data.pkl', 'wb') as f:
            pickle.dump(data_to_save, f)

        print('executed %.2f s, saved data for %.2f s' %
            (end-start, len(data_to_save)/fps))




def decode_data(data, frame_queue, message_queue, ignore=[], print_flag=1):
    # print('decoding')
    raw_data = data[:]

    # Q7I = one Q (unsigned long long) and seven Is (unsigned int)
    try:
        magic, version, length, platform, frameNum, cpuCycles, numObj, numTLVs = struct.unpack(
            'Q7I', data[:header_size])
    except struct.error:
        print('Failed decoding hearder')
        return

    if print_flag:
        # os.system('clear')
        print("Packet ID:\t%d " % (frameNum))
        print("Packet len:\t%d " % (length))
    # print("TLV:\t\t%d "%(numTLVs))
    # print("Detect Obj:\t%d "%(numObj))

    if numTLVs > 1024:
        return

    data = data[header_size:]
    xs = []
    ys = []
    zs = []
    dopplers = []
    ranges = []
    peaks = []

    for i in range(numTLVs):
        tlvType, tlvLength = struct.unpack('2I', data[:8])
        data = data[8:]
        if (tlvType == 1):
            xs, ys, zs, dopplers, ranges, peaks = parseDetectedObjects(
                data, tlvLength, ignore, print_flag)
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

    return xs, ys, zs, dopplers, ranges, peaks


def parseDetectedObjects(data, tlvLength, ignore=[], print_flag=1):
    # hreader = two unsigned short
    numDetectedObj, xyzQFormat = struct.unpack('2H', data[:4])
    xs = []
    ys = []
    zs = []
    dopplers = []
    ranges = []
    peaks = []
    # print("\tDetect Obj:\t%d "%(numDetectedObj))
    for i in range(numDetectedObj):
        # print("\tObjId:\t%d "%(i))
        # each object = 6 short, 1st and 3rd being unsigned
        rangeIdx, dopplerIdx, peakVal, x, y, z = struct.unpack(
            'HhH3h', data[4+12*i:4+12*i+12])
        x = (x*1.0/(1 << xyzQFormat))
        y = (y*1.0/(1 << xyzQFormat))
        z = (z*1.0/(1 << xyzQFormat))

        condition = y > 0 and y < 10
        if condition is True:
            if print_flag:
                print("\t\tDopplerIdx:\t%d " % (dopplerIdx))
                print("\t\tRangeIdx:\t%d " % (rangeIdx))
                print("\t\tPeakVal:\t%d " % (peakVal))
                print("\t\tX (left-right):\t\t%07.3f " % (x))
                print("\t\tY (depth):\t\t%07.3f " % (y))
                print("\t\tZ (up-down):\t\t%07.3f " % (z))
                print()
                # print("%07.3f "%(y))
            xs.append(x)
            ys.append(y)
            zs.append(z)
            peaks.append(peakVal)
            dopplers.append(dopplerIdx)
            ranges.append(rangeIdx)
    return xs, ys, zs, dopplers, ranges, peaks


def parseRangeProfile(data, tlvLength):
    for i in range(256):
        rangeProfile = struct.unpack('H', data[2*i:2*i+2])
        print("\tRangeProf[%d]:\t%07.3f " %
              (i, rangeProfile[0] * 1.0 * 6 / 8 / (1 << 8)))
    print("\tTLVType:\t%d " % (2))


def parseStats(data, tlvLength):
    interProcess, transmitOut, frameMargin, chirpMargin, activeCPULoad, interCPULoad = struct.unpack(
        '6I', data[:24])
    print("\tOutputMsgStats:\t%d " % (6))
    print("\t\tChirpMargin:\t%d " % (chirpMargin))
    print("\t\tFrameMargin:\t%d " % (frameMargin))
    print("\t\tInterCPULoad:\t%d " % (interCPULoad))
    print("\t\tActiveCPULoad:\t%d " % (activeCPULoad))
    print("\t\tTransmitOut:\t%d " % (transmitOut))
    print("\t\tInterprocess:\t%d " % (interProcess))


def vis_thread(frame_queue, message_queue, runflag):
    camera = 1 if device[1] == 'IT070107' else 0
    cv2.namedWindow('win')
    cv2.namedWindow('background removed')
    vc = cv2.VideoCapture(camera)
    rval, frame = vc.read()
    h = frame.shape[0]
    w = frame.shape[1]

    if not vc.isOpened():  # try to get the first frame
        print('camera not found')
        runflag.value = 0
        return
    
    FoV_h = 28
    FoV_v = 28
    
    mode = 'init'
    start = time.time()
    background_x = []
    background_y = []
    background_z = []
    model = None
    n_cluster = 1
    colors = []


    while runflag.value == 1:
        FoVrh = FoV_h/180*np.pi
        FoVrv = FoV_v/180*np.pi

        rval, frame = vc.read()
        # frame = cv2.flip(frame, 1)
        obj = frame.copy()

        try:
            cmd, xs, ys, zs, peaks, ranges = frame_queue.get(block=True, timeout=3)
        except queue.Empty:
            print('Queue Empty')
            runflag.value = 0
            break

        if len(xs) == 0:
            continue 

        data = np.stack((xs, ys, zs), axis=-1)
        if mode == 'run':
            labels = model.fit_predict(data)
            

        for i in range(len(xs)):
            x, y, z = data[i]
            # x = int((xs[i]+0.5)*w)
            # z = int((zs[i]+0.5)*h)
            # peak = peaks[i]
            angle_h = np.arctan(x/y)
            angle_v = -np.arctan(z/y)
            range_idx = ranges[i]
            if range_idx >= fft_distance.shape[0] or fft_distance[range_idx] < 0.3:
                continue
            r = int(500/range_idx)

            x = int((angle_h/FoVrh + 1) * 0.5*w)
            z = int((angle_v/FoVrv + 1) * 0.5*h)

            n = y/2*255

            color = (0, 255, 255)
            cv2.circle(frame, (x, z), r, color, -1)

            if mode == 'run':
                class_idx = labels[i]
                if class_idx == -1:
                    cv2.circle(obj, (x, z), r, (0, 255, 255), -1)
                else:
                    cv2.circle(obj, (x, z), r, colors[class_idx], -1)

            
        cv2.imshow('win', frame)
        cv2.imshow('background removed', obj)

        plt.clf()
        plt.hist(fft_distance[ranges], 256, [0, 10])
        plt.ylim([0, 20])
        plt.pause(0.005)

        key = cv2.waitKey(1)
        if key == 27:
            runflag.value = 0
            break

        if key == ord('-'):
            FoV_v -= 1
            print('Field of View changed to', FoV_v)
        if key == ord('='):
            FoV_v += 1
            print('Field of View changed to', FoV_v)

        if key == ord('d'):
            my_debug(frame, data, labels, colors)

        if mode == 'init':
            background_x += xs
            background_y += ys
            background_z += zs
            if time.time() - start > 10:
                mode = 'run'
                data = np.stack(
                    (background_x, background_y, background_z), axis=-1)

                model = cluster(data)
                label = model.labels_
                n_cluster = np.unique(label).shape[0]
                colors = get_random_colors(n_cluster-1)
                print(np.unique(label), n_cluster)
                

    # plt.show()

def my_debug(frame, data, labels, colors):
    pdb.set_trace()


def cluster(data):
    model = DBSCAN(eps=0.1)
    model.fit((data))
    return model


def data_thread(frame_queue, message_queue, runflag):
    cfg = read_cfg()
    send_cfg(cfg, frame_queue, message_queue, runflag)


def get_spaced_colors(n):
    max_value = 16581375  # 255**3
    interval = int(max_value / n)
    colors = [hex(I)[2:].zfill(6) for I in range(0, max_value, interval)]
    return [(int(i[:2], 16), int(i[2:4], 16), int(i[4:], 16)) for i in colors]

def get_random_colors(n):
    return [(np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256)) for _ in range(n)]

def main():
    runflag = multiprocessing.Value('i', 1)
    frame_queue = multiprocessing.Queue()
    message_queue = multiprocessing.Queue()
    t0 = multiprocessing.Process(target=vis_thread, args=(
        frame_queue, message_queue, runflag, ))
    t1 = multiprocessing.Process(target=data_thread, args=(
        frame_queue, message_queue, runflag, ))
    t0.start()
    t1.start()
    t0.join()
    t1.join()


if __name__ == '__main__':
    main()
