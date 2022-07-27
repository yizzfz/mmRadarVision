import serial
import time
import pickle
import struct
import numpy as np
import sys

magic_word = b'\x02\x01\x04\x03\x06\x05\x08\x07'


class Radar():
    def __init__(self, name, cfg_port, data_port, runflag, studio_cli_image=False, debug=False, outformat='p', sdk=None):
        """
        Parameters:
            name: String, '1443', '1642, '1843', etc.
            cfg_port: Int, Application/User COM port.
            data_port: Int, Data COM port.
            runflag: Global flag, control the status of the system.
            sdk: Float, if a specific sdk version is used. Default None.
            outformat: 'p' for pointcloud, 'v' for velocity, 's' for snr, 'n' for noise. Can be 'pvsn' for example. 
            studio_cli_image: Bool, if the radar is loaded with a studio_cli image.
            debug: Bool, print debug information.
        """
        self.name = name
        self.cfg_port_name = cfg_port
        self.data_port_name = data_port
        self.cfg_port = None
        self.data_port = None
        self.runflag = runflag
        self.studio_cli_image = studio_cli_image
        self.debug = debug
        oldsdk = False
        self.side_info = False
        if sdk is None:
            if '1443' in self.name or '1642' in self.name:
                oldsdk = True
        elif sdk < 3:
            oldsdk = True

        if oldsdk:
            self.log('Assuming the old SDK is used (version < 3.0).')
            if outformat is not 'p':
                self.log('Ignoring output format')
                outformat = 'p'
            self.decode_func = self.parse_detected_objects_oldsdk
        else:
            self.decode_func = self.parse_detected_objects
            if 's' in outformat or 'n' in outformat:
                self.side_info = True
        self.outformat = self.outformat_mask(outformat)

    def connect(self, cfg_file) -> bool:
        cfg = read_cfg(cfg_file)
        try:
            speed = 115200 if not self.studio_cli_image else 921600
            cfg_port = serial.Serial(self.cfg_port_name, speed, timeout=5)
            data_port = serial.Serial(
                self.data_port_name, 921600, timeout=0.01)
            data_port.set_buffer_size(rx_size=128000)
        except serial.serialutil.SerialException:
            self.log('Failed opening serial port, check connection')
            self.runflag.value = 0
            return False

        assert cfg_port.is_open and data_port.is_open
        self.cfg_port = cfg_port
        self.data_port = data_port

        self.log('connected')
        for line in cfg:
            # self.log(f'send {line}')
            line = (line+'\n').encode()
            cfg_port.write(line)
            time.sleep(0.05)

            try:
                full_message = ''
                while cfg_port.inWaiting():
                    message = cfg_port.read(cfg_port.inWaiting()).decode()
                    full_message += message
                    
                if 'error' in full_message.lower():
                    self.log('cfg error')
                    self.log(full_message)
                    self.runflag.value = 0
                    return False

            except UnicodeDecodeError as e:
                self.log(e.reason)
                self.log(full_message)
                return False

        self.log('all cfg sent')
        return True

    def run_periodically(self, frame_queue, period=3):
        start = time.time()
        run = True
        data = b''
        send = 0
        while self.runflag.value == 1:
            if time.time() - start > period:
                run = not run
                start = time.time()
                if run:
                    data = b''
                    send = 0
                    self.cfg_port.write('sensorStart\n'.encode())
                    self.log('start')
                else:
                    self.cfg_port.write('sensorStop\n'.encode())
                    self.log('stop')

            if run:
                data_line = self.data_port.read(32)
                if magic_word in data_line:
                    if data != b'' and send:
                        out = self.decode_data(data, frame_queue)
                    data = b''
                    send = 1
                data += data_line
            else:
                if frame_queue.empty():
                    frame_queue.put(([]))
        self.cfg_port.write('sensorStop\n'.encode())


    def run(self, frame_queue):
        # self.clear_cmd()
        self.log('sensor start')
        self.cfg_port.write('sensorStart\n'.encode())
        data = b''
        send = 0
        try:
            while self.runflag.value == 1:
                data_line = self.data_port.read(8)
                if self.debug:
                    self.log(f'Received {len(data_line)} bytes')
                if self.studio_cli_image:
                    continue
                if magic_word in data_line:
                    assert(data_line.startswith(magic_word))
                    if data != b'' and send:
                        out = self.decode_data(data, frame_queue)
                    data = b''
                    send = 1
                data += data_line
        except KeyboardInterrupt:
            pass
        self.cfg_port.write('sensorStop\n'.encode())
        self.log('sensor stop')
        time.sleep(0.5)
        # self.clear_cmd()
        self.exit()

    def clear_cmd(self):
        while self.cfg_port.inWaiting():
            self.cfg_port.read(self.cfg_port.inWaiting())

    def test(self):
        self.cfg_port.write('sensorStart\n'.encode())
        data = b''
        send = 0
        while 1:
            bytes_to_read = self.data_port.in_waiting
            print(bytes_to_read)
            data_line = self.data_port.read(64)
            if magic_word in data_line:
                if not data_line.startswith(magic_word):
                    print(data_line)
                if data != b'' and send:
                    print(len(data))
                data = b''
                send = 1
            data += data_line
        self.cfg_port.write('sensorStop\n'.encode())
      

    def decode_data(self, data, frame_queue):
        if self.debug:
            self.log(f'Decoding data packet of size {len(data)}')

        # Q7I = one Q (unsigned long long) and seven Is (unsigned int)
        try:
            if '1443' in self.name:
                header_size = 8 + 4 * 7
                magic, version, length, platform, frameNum, cpuCycles, numObj, numTLVs = struct.unpack(
                    'Q7I', data[:header_size])
            else:
                header_size = 8 + 4 * 8
                magic, version, length, platform, frameNum, cpuCycles, numObj, numTLVs, subFrameNumber = struct.unpack(
                    'Q8I', data[:header_size])
        except struct.error:
            self.log('Failed decoding header') 
            return None

        if self.debug:
            self.log("Packet ID:\t%d " % (frameNum))
            self.log("Packet len:\t%d " % (length))

        if numTLVs > 1024:
            return

        data = data[header_size:]
        res = None
        side = None

        for i in range(numTLVs):
            try:
                tlvType, tlvLength = struct.unpack('2I', data[:8])
            except struct.error:
                self.log('Failed decoding TLV')
                return None
            data = data[8:]
            if (tlvType == 1):      # DETECTED_POINTS
                res = self.decode_func(data, tlvLength)
            elif (tlvType == 7) and self.side_info:    # DETECTED_POINTS_SIDE_INFO
                side = self.parse_side_info(data, tlvLength)
            elif (tlvType == 2):    # RANGE_PROFILE
                res = self.parseRangeProfile(data, tlvLength)
            # elif (tlvType == 6):
            #     parseStats(data, tlvLength)
            else:
                self.log("tlv type %d not implemented" % (tlvType))
            data = data[tlvLength:]

        if frame_queue.empty() and res is not None:
            if side is not None:
                res = np.stack((res, side), axis=1)
            res = res[:, self.outformat]
            frame_queue.put((res))
        return res
       
    def log(self, txt):
        print(f'[{self.name}] {txt}')

    def parse_side_info(self, data, tlvLength):
        assert (tlvLength % 4 == 0)        # each point has 2 2-byte words
        numDetectedObj = int(tlvLength/4)
        snrs = []
        noises = []
        for i in range(numDetectedObj):
            # each object = 2 int16
            try:
                snr, noise = struct.unpack(
                    '2h', data[4*i:4*i+4])
            except struct.error:
                self.log('Failed decoding side info')
                return None
            snrs.append(snr)
            noises.append(noise)

        res = np.stack((snrs, noises), axis=1)
        return res


    def parse_detected_objects(self, data, tlvLength):
        assert (tlvLength % 16 == 0)        # each point has 4 4-byte words
        numDetectedObj = int(tlvLength/16)
        xs = []
        ys = []
        zs = []
        vs = []

        # self.log("\tDetect Obj:\t%d "%(numDetectedObj))
        for i in range(numDetectedObj):
            # self.log("\tObjId:\t%d "%(i))
            # each object = 4 floats (16 bytes)
            try:
                x, y, z, v = struct.unpack(
                    '4f', data[16*i:16*i+16])
            except struct.error:
                self.log('Failed decoding object')
                return None

            condition = y > 0.2 and y < 10
            if condition is True:
                if self.debug:
                    # self.log("\t\tDopplerIdx:\t%d " % (dopplerIdx))
                    # self.log("\t\tRangeIdx:\t%d " % (rangeIdx))
                    # self.log("\t\tPeakVal:\t%d " % (peakVal))
                    self.log("\t\tX (left-right):\t\t%07.3f " % (x))
                    self.log("\t\tY (depth):\t\t%07.3f " % (y))
                    self.log("\t\tZ (up-down):\t\t%07.3f " % (z))
                    self.log()
                    # self.log("%07.3f "%(y))
                xs.append(x)
                ys.append(y)
                zs.append(z)
                vs.append(v)
            res = np.stack((xs, ys, zs, vs), axis=1)
        return res


    def parse_detected_objects_oldsdk(self, data, tlvLength):
        # header = two unsigned short
        numDetectedObj, xyzQFormat = struct.unpack('2H', data[:4])
        xs = []
        ys = []
        zs = []
        dopplers = []
        ranges = []
        peaks = []
        # self.log("\tDetect Obj:\t%d "%(numDetectedObj))
        for i in range(numDetectedObj):
            # self.log("\tObjId:\t%d "%(i))
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
                if self.debug:
                    self.log("\t\tDopplerIdx:\t%d " % (dopplerIdx))
                    self.log("\t\tRangeIdx:\t%d " % (rangeIdx))
                    self.log("\t\tPeakVal:\t%d " % (peakVal))
                    self.log("\t\tX (left-right):\t\t%07.3f " % (x))
                    self.log("\t\tY (depth):\t\t%07.3f " % (y))
                    self.log("\t\tZ (up-down):\t\t%07.3f " % (z))
                    self.log()
                    # self.log("%07.3f "%(y))
                xs.append(x)
                ys.append(y)
                zs.append(z)
                peaks.append(peakVal)
                dopplers.append(dopplerIdx)
                ranges.append(rangeIdx)

        # xs, ys, zs, dopplers, ranges, peaks
        return np.stack((xs, ys, zs), axis=1)

    def outformat_mask(self, outformat):
        m = []
        if 'p' in outformat:
            m += [0, 1, 2]
        if 'v' in outformat:
            m.append(3)
        if 's' in outformat:
            m.append(4)
        if 'n' in outformat:
            m.append(5)

    def exit(self):
        self.cfg_port.close()
        self.data_port.close()
        self.log('All ports closed')


# https://e2e.ti.com/support/sensors/f/1023/p/830398/3073763
# Range profile
# Type: (MMWDEMO_OUTPUT_MSG_RANGE_PROFILE)
# Length: (Range FFT size) x(size of uint16_t)
# Value: Array of profile points at 0th Doppler(stationary objects).
# The points represent the sum of log2 magnitudes of received antennas expressed in Q9 format.
    def parseRangeProfile(self, data, tlvLength):
        try:
            res = np.asarray(struct.unpack('256H', data[:512]))
        except struct.error:
            self.log('Failed decoding range profile')
            return None
        return res

        


def read_cfg(file):
    cfg = []
    with open(file) as f:
        lines = f.read().split('\n')
    for line in lines:
        if not line.startswith('%'):
            cfg.append(line)
    return cfg


if __name__ == '__main__':
    r = Radar('test', 'COM6', 'COM5', 1)
    r.connect('../iwr1443/cfg/new.cfg')
    r.test()

"""
TLV type SKD >= 3.0
MMWDEMO_OUTPUT_MSG_DETECTED_POINTS = 1,
MMWDEMO_OUTPUT_MSG_RANGE_PROFILE,
MMWDEMO_OUTPUT_MSG_NOISE_PROFILE,
MMWDEMO_OUTPUT_MSG_AZIMUT_STATIC_HEAT_MAP,
MMWDEMO_OUTPUT_MSG_RANGE_DOPPLER_HEAT_MAP,
MMWDEMO_OUTPUT_MSG_STATS,
MMWDEMO_OUTPUT_MSG_DETECTED_POINTS_SIDE_INFO,
MMWDEMO_OUTPUT_MSG_AZIMUT_ELEVATION_STATIC_HEAT_MAP,
MMWDEMO_OUTPUT_MSG_TEMPERATURE_STATS,

TLV type SKD < 3.0
MMWDEMO_OUTPUT_MSG_DETECTED_POINTS = 1,
MMWDEMO_OUTPUT_MSG_RANGE_PROFILE,
MMWDEMO_OUTPUT_MSG_NOISE_PROFILE,
MMWDEMO_OUTPUT_MSG_AZIMUT_STATIC_HEAT_MAP,
MMWDEMO_OUTPUT_MSG_RANGE_DOPPLER_HEAT_MAP,
MMWDEMO_OUTPUT_MSG_STATS,
"""