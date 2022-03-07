import cv2
import serial
import time
import struct
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

'''
C:/ti/mmwave_sdk_01_01_00_02/packages/ti/demo/xwr16xx/mmw/docs/doxygen/html/struct_mmw_demo__output__message__header__t.html
'''


magic_word = b'\x02\x01\x04\x03\x06\x05\x08\x07'
header_size = 8 + 4 * 7

def read_cfg():
    cfg = []
    with open('cfg/zoneA.cfg') as f:
        lines = f.read().split('\n')
    for line in lines:
        if not line.startswith('%'):
            cfg.append(line)
    return cfg


def send_cfg(cfg):
    try:
        cfg_port = serial.Serial('COM6', 115200)
        data_port = serial.Serial('COM7', 921600, timeout=0.01)
    except serial.serialutil.SerialException:
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
    while 1:
        cnt += 1
        bytes_to_read = data_port.in_waiting
        print(bytes_to_read)
        data_line = data_port.read(32)
        if magic_word in data_line:
            if data != b'' and send:
                decode_data(data, ignore)
            data = b''
            send = 1
        data += data_line
        # buf = data_port.read(32)
        # cur = 0
        # while(cur < len(buf)):
            # data_line = buf[cur:cur+32]
            # cur += 32
            # if magic_word in data_line:
            #     if data != b'' and send:
            #         decode_data(data, ignore, cnt%5==0)
            #     data = b''
            #     send = 1
            # data += data_line


def decode_data(data, ignore=[], print_flag=1):
    # print('decoding')
    raw_data = data[:]

    # Q7I = one Q (unsigned long long) and seven Is (unsigned int)
    try:
        magic, version, length, platform, frameNum, cpuCycles, numObj, numTLVs = struct.unpack('Q7I', data[:header_size])
    except struct.error:
        print ('Failed decoding hearder')
        return

    if print_flag:
        os.system('clear')
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

    for i in range(numTLVs):
        tlvType, tlvLength = struct.unpack('2I', data[:8])
        data = data[8:]
        if (tlvType == 1):
            xs, ys, zs, doppler, ranges = parseDetectedObjects(data, tlvLength, ignore, print_flag)
        elif (tlvType == 2):
            parseRangeProfile(data, tlvLength)
        elif (tlvType == 6):
            parseStats(data, tlvLength)
        else:
            print("Unidentified tlv type %d"%(tlvType))
        data = data[tlvLength:]

    # if frameNum < 100:
    #     for i in range(len(xs)):
    #         ignore.append((xs[i], ys[i], zs[i]))
    # if frameNum == 100:
    #     import pdb; pdb.set_trace()

    plt.clf()
    ax = fig.add_subplot(211, projection='3d')
    ax.scatter(xs, ys, zs)
    ax.set_xlim([-5, 5])
    ax.set_ylim([-5, 5])
    ax.set_zlim([-5, 5])
    ax2 = fig.add_subplot(212)
    ax2.hist(ys)
    ax2.set_xlim([0.01, 0.5])
    ax2.set_ylim([0, 5])
    plt.pause(0.05)
    return


def parseDetectedObjects(data, tlvLength, ignore=[], print_flag=1):
    # hreader = two unsigned short
    numDetectedObj, xyzQFormat = struct.unpack('2H', data[:4])
    xs = []
    ys = []
    zs = []
    doppler = []
    ranges = []
    # print("\tDetect Obj:\t%d "%(numDetectedObj))
    for i in range(numDetectedObj):
        # print("\tObjId:\t%d "%(i))
        # each object = 6 short, 1st and 3rd being unsigned
        rangeIdx, dopplerIdx, peakVal, x, y, z = struct.unpack('HhH3h', data[4+12*i:4+12*i+12])
        x = (x*1.0/(1 << xyzQFormat))
        y = (y*1.0/(1 << xyzQFormat))
        z = (z*1.0/(1 << xyzQFormat))

        condition = rangeIdx > 10 and rangeIdx < 100
        condition = (x, y, z) not in ignore
        condition = abs(z) < 0.5 and abs(x)<0.5 and abs(y)<3
        condition = True
        if condition is True:
            if print_flag:
                print("\t\tDopplerIdx:\t%d "%(dopplerIdx))
                print("\t\tRangeIdx:\t%d "%(rangeIdx))
                print("\t\tPeakVal:\t%d "%(peakVal))
                print("\t\tX (left-right):\t\t%07.3f "%(x))
                print("\t\tY (depth):\t\t%07.3f "%(y))
                print("\t\tZ (up-down):\t\t%07.3f "%(z))
                print("%07.3f "%(y))
            xs.append(x)
            ys.append(y)
            zs.append(z)
            doppler.append(dopplerIdx)
            ranges.append(rangeIdx)
    return xs, ys, zs, doppler, ranges


def parseRangeProfile(data, tlvLength):
    # data = struct.unpack('256H', data[:256])
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


def main():


    cfg = read_cfg()
    send_cfg(cfg)

    plt.show()



if __name__ == '__main__':
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    main()
