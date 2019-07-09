import serial
import time
import pickle
import struct
import numpy as np

magic_word = b'\x02\x01\x04\x03\x06\x05\x08\x07'


class Radar():
    def __init__(self, name, cfg_port, data_port, save=None):
        self.name = name
        self.cfg_port_name = cfg_port
        self.data_port_name = data_port
        self.save = save

    def start(self, cfg_file, frame_queue, runflag):
        self.run = runflag
        cfg = read_cfg(cfg_file)
        try:
            cfg_port = serial.Serial(self.cfg_port_name, 115200)
            data_port = serial.Serial(
                self.data_port_name, 921600, timeout=0.01)
        except serial.serialutil.SerialException:
            self.log('Failed opening serial port, check connection')
            self.run.value = 0
            return -1

        assert cfg_port.is_open and data_port.is_open

        self.log('connected')
        for line in cfg:
            # self.log('[send]', line)
            line = (line+'\n').encode()
            cfg_port.write(line)
            time.sleep(0.05)

            res = ''
            while(res == '' or cfg_port.inWaiting()):
                read_len = cfg_port.inWaiting()
                res += cfg_port.read(read_len).decode()
            # self.log(res, end='\n\n')
            if 'Error' in res:
                self.log('cfg error')
                self.log(res)
                self.run.value = 0
                return -1

        data = b''
        send = 0
        self.log('all cfg sent')

        data_to_save = []
        start = time.time()
        while self.run.value == 1:
            # bytes_to_read = data_port.in_waiting
            # print(bytes_to_read)
            data_line = data_port.read(32)
            if magic_word in data_line:
                if data != b'' and send:
                    # enable print flag?
                    out = self.decode_data(data, frame_queue, 0)
                    if time.time() - start > 30 and time.time() - start < 330:
                        data_to_save.append(out)

                data = b''
                send = 1
            data += data_line

        cfg_port.write('sensorStop\n'.encode())
        end = time.time()

        if self.save:
            with open('data.pkl', 'wb') as f:
                pickle.dump(data_to_save, f)


    def decode_data(self, data, frame_queue, print_flag=1):
        # print('decoding')
        raw_data = data[:]

        # Q7I = one Q (unsigned long long) and seven Is (unsigned int)
        try:
            if '1443' in self.name:
                header_size = 8 + 4 * 7
                magic, version, length, platform, frameNum, cpuCycles, numObj, numTLVs = struct.unpack(
                    'Q7I', data[:header_size])
            else:
                header_size = 8 + 4 * 8
                magic, version, length, platform, frameNum, cpuCycles, numObj, numTLVs, _ = struct.unpack(
                    'Q8I', data[:header_size])
        except struct.error:
            self.log('Failed decoding hearder')
            return None

        if print_flag:
            # os.system('clear')
            print("Packet ID:\t%d " % (frameNum))
            print("Packet len:\t%d " % (length))

        if numTLVs > 1024:
            return

        data = data[header_size:]
        res = None

        for i in range(numTLVs):
            try:
                tlvType, tlvLength = struct.unpack('2I', data[:8])
            except struct.error:
                self.log('Failed decoding TLV')
                return None
            data = data[8:]
            if (tlvType == 1):
                if '6843' in self.name:
                    res = self.parseDetectedObjects6843(
                        data, tlvLength, print_flag)
                else:
                    res = self.parseDetectedObjects(
                        data, tlvLength, print_flag)

            elif (tlvType == 7):
                pass
            # elif (tlvType == 2):
            #     parseRangeProfile(data, tlvLength)
            # elif (tlvType == 6):
            #     parseStats(data, tlvLength)

            else:
                self.log("tlv type %d not implemented" % (tlvType))
            data = data[tlvLength:]

        if frame_queue.empty():
            frame_queue.put((res))

        return res
       
    def log(self, txt):
        print(f'[{self.name}] {txt}')

    def parseDetectedObjects6843(self, data, tlvLength, print_flag=1):
        assert (tlvLength % 16 == 0)
        numDetectedObj = int(tlvLength/16)
        xs = []
        ys = []
        zs = []
        vs = []

        # print("\tDetect Obj:\t%d "%(numDetectedObj))
        for i in range(numDetectedObj):
            # print("\tObjId:\t%d "%(i))
            # each object = 4 floats (16 bytes)
            try:
                x, y, z, v = struct.unpack(
                    '4f', data[16*i:16*i+16])
            except struct.error:
                self.log('Failed decoding object')
                return None

            condition = y > 0.2 and y < 10
            if condition is True:
                if print_flag:
                    # print("\t\tDopplerIdx:\t%d " % (dopplerIdx))
                    # print("\t\tRangeIdx:\t%d " % (rangeIdx))
                    # print("\t\tPeakVal:\t%d " % (peakVal))
                    print("\t\tX (left-right):\t\t%07.3f " % (x))
                    print("\t\tY (depth):\t\t%07.3f " % (y))
                    print("\t\tZ (up-down):\t\t%07.3f " % (z))
                    print()
                    # print("%07.3f "%(y))
                xs.append(x)
                ys.append(y)
                zs.append(z)

        return np.stack((xs, ys, zs), axis=1)


    def parseDetectedObjects(self, data, tlvLength, print_flag=1):
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
            try:
                rangeIdx, dopplerIdx, peakVal, x, y, z = struct.unpack(
                    'HhH3h', data[4+12*i:4+12*i+12])
            except struct.error:
                self.log('Failed decoding object')
                return None

            x = (x*1.0/(1 << xyzQFormat))
            y = (y*1.0/(1 << xyzQFormat))
            z = (z*1.0/(1 << xyzQFormat))

            condition = y > 0.2 and y < 10
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

        # xs, ys, zs, dopplers, ranges, peaks
        return np.stack((xs, ys, zs), axis=1)
        


def read_cfg(file):
    cfg = []
    with open(file) as f:
        lines = f.read().split('\n')
    for line in lines:
        if not line.startswith('%'):
            cfg.append(line)
    return cfg
