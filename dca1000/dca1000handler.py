import json
import os
import subprocess
import time
import numpy as np
import sys
import socket
from threading import Thread
from collections import deque
from queue import Empty
sys.path.insert(1, 'C:/Users/hc13414/OneDrive - University of Bristol/mmwave/simulation/algo')
from fft_processor import FFTProcessor

dependency = ['DCA1000EVM_CLI_Control.exe', 'DCA1000EVM_CLI_Record.exe', 'RF_API.dll']

"""Only designed with 1 RX"""
class DCA1000Handler:
    def __init__(self, model, radarcfg, runflag, queue, send_rate=1, port=60203, data_format='fft'):
        self.cwd = os.path.dirname(os.path.abspath(__file__))
        self.runflag = runflag
        self.model = model
        self.port = port
        self.data_format = data_format
        self.Q = queue
        self.Q.cancel_join_thread()
        self.steps = 0
        self.samples_per_chirp = 0
        self.frames_per_second = 0
        self.chirps_per_frame = 0
        # self.data_per_second = 0
        # self.bytes_per_second = 0
        self.data_per_sp = 0
        self.bytes_per_sp = 0
        self.send_rate = send_rate
        self.cfg_name = 'tmp.json'
        for f in dependency:
            assert os.path.exists(os.path.join(self.cwd, f)) and f"{f} is required to run DCA1000"
        self.parse_radarcfg(radarcfg)
        # self.data_epilog = None
        assert self.data_per_sp > 0 and "Failed to read radar cfg"

        jsonfile = 'dca1000.json'
        with open(os.path.join(self.cwd, jsonfile)) as f:
            dca1000config = json.load(f)
        dca1000config = self.convert_to_abs_path(dca1000config)
        if '1642' in model or '6843' in model or '1843' in model:
            dca1000config['DCA1000Config']['lvdsMode'] = 2
        else:
            raise ValueError(f'radar not supported')
        dca1000config['DCA1000Config']['captureConfig']['filePrefix'] = model
        self.data_location = dca1000config['DCA1000Config']['captureConfig']['fileBasePath']
        self.data_prefix = dca1000config['DCA1000Config']['captureConfig']['filePrefix']
        # self.data_size = dca1000config['DCA1000Config']['captureConfig']['maxRecFileSize_MB'] * 1e6       # in bytes
        self.log(f'Expecting {self.data_per_sp * self.send_rate:.0f} bytes of data per seconds')
        # self.log(f'Each file should contain {self.data_size/self.bytes_per_second:.2f} seconds of data')

        self.FP = FFTProcessor(radarcfg, multiplier=4, max_d=5)
        with open(os.path.join(self.cwd, 'tmp.json'), 'w') as write_file:
            json.dump(dca1000config, write_file, indent=2)
        self.control_exe = os.path.join(self.cwd, 'DCA1000EVM_CLI_Control.exe')
        try:
            # self.run_cmd('reset_fpga')
            # self.run_cmd('reset_ar_device')
            self.run_cmd('fpga')
            self.run_cmd('record')
        except RuntimeError as e:
            self.log('Configuration of DCA1000 failed')
            self.log(e)
            self.runflag.value = 0

        self.t = Thread(target=self.run_server, args=())
        self.t.daemon = True
        self.t.start()

    def receive_all(self, socket):
        data = bytearray()
        while len(data) < self.bytes_per_sp:
            packet = socket.recv(self.bytes_per_sp - len(data))
            if not packet:
                raise ValueError('No data received')
            data.extend(packet)
        return data

    def adc_format(self, adcData):
        if '1642' in self.model or '6843' in self.model or '1843' in self.model:
            adcData = adcData.reshape((-1, 4))
            adcData = adcData[:, 0:2] + 1j*adcData[:, 2:4]
        else:
            raise ValueError('Radar model incorrect')
        return adcData

    def run_server(self):
        first_packet = True
        t0 = None
        data_rate = deque(maxlen=10)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('127.0.0.1', self.port))
            self.log(f'Server listening at port {self.port}')
            s.listen()
            conn, addr = s.accept()
            self.log(f"Connected by {addr}")
            with conn:
                try:
                    while self.runflag.value == True:
                        data = self.receive_all(conn)
                        adcData = np.frombuffer(data, dtype=np.int16)
                        adcData = self.adc_format(adcData)
                        adcData = adcData.reshape(self.data_sp_shape)
                        if self.data_format == 'fft':
                            fftData = self.FP.compute_FFTs(adcData, split=False)
                            self.Q.put(fftData)
                        elif self.data_format == 'raw':
                            self.Q.put(adcData)
                        t1 = time.time()
                        if t0 is not None:
                            data_rate.append(t1-t0)
                            congestion = (np.mean(data_rate)*self.send_rate-1)*100
                            if len(data_rate) == 10 and congestion > 3:
                                self.log(
                                    f'Warning: data congestion detected, rate {congestion:.2f}%')
                        t0 = t1
                        if first_packet:
                            first_packet = False
                            datashape = fftData[0].shape if self.data_format == 'fft' else adcData.shape
                            self.log(f'Data transmission established successfully, data shape {datashape}')
                except (Exception, KeyboardInterrupt) as e:
                    self.log(e)
                    self.runflag.value=False
        self.Q.close()

    def run(self):
        self.clean_files()
        self.run_cmd('start_record')
        self.log('Recording starts')
        try:
            while self.runflag.value == 1:
                # self.decode_datafile()
                time.sleep(0.1)
        except KeyboardInterrupt:
            pass
        self.run_cmd('stop_record')
        self.exit()

    def exit(self):
        self.t.join()
        # self.clean_files()
        self.log('Exited')

    def parse_radarcfg(self, config):
        self.samples_per_chirp = config['samples_per_chirp']
        self.chirps_per_frame = config['chirps_per_frame']
        self.frames_per_second = int(1/config['frame_time'])
        # self.data_per_second = self.samples_per_chirp * self.chirps_per_frame * self.frames_per_second
        # self.bytes_per_second = self.data_per_second * 4
        # self.data_second_shape = (self.frames_per_second * self.chirps_per_frame, self.samples_per_chirp)

        self.frame_per_sp = int(self.frames_per_second/self.send_rate)
        self.data_per_sp = self.samples_per_chirp * self.chirps_per_frame * self.frame_per_sp
        self.bytes_per_sp = self.data_per_sp * 4
        self.data_sp_shape = self.frame_per_sp * self.chirps_per_frame, self.samples_per_chirp

    # def read_bin(self, filename):
    #     adcData = np.fromfile(filename, dtype=np.int16)
    #     adcData = adcData.reshape((-1, 2)).T
    #     adcData = adcData[0] + 1j*adcData[1]
    #     if self.data_epilog is not None:
    #         adcData = np.concatenate((self.data_epilog, adcData))
    #     while adcData.shape[0] > self.data_per_second:
    #         data_block = adcData[:self.data_per_second]
    #         data_block = data_block.reshape(self.data_second_shape)
    #         self.Q.put(self.FP.compute_FFTs(data_block))
    #         adcData = adcData[self.data_per_second:]
    #     self.data_epilog = adcData

    def clean_files(self):
        cnt = 0
        if not os.path.isdir(self.data_location):
            return
        for root, _, files in os.walk(self.data_location):
            for f in files:
                if f.startswith(f'{self.data_prefix}_Raw_') and f.endswith('.bin'):
                    os.remove(os.path.join(root, f))
                    cnt += 1
        self.log(f'cleaned {cnt} data files')

    # def decode_datafile(self):
    #     datafile = f'{self.data_prefix}_Raw_{self.steps}.bin'
    #     datafile = os.path.join(self.data_location, datafile)
    #     if not os.path.exists(datafile) or os.path.getsize(datafile) < self.data_size:
    #         time.sleep(0.5)
    #         return
    #     self.log(f'found {datafile}')
    #     self.read_bin(datafile)
    #     self.steps += 1

    def run_cmd(self, cmd=''):
        # exe = self.record_exe if cmd in ['start_record', 'stop_record'] else self.control_exe
        res = subprocess.run([self.control_exe, cmd, self.cfg_name], cwd=self.cwd)
        returncode = res.returncode
        if returncode != 0:
            self.runflag.value = 0
            raise RuntimeError(f'Failed to send cmd {cmd} to DCA1000')

    def convert_to_abs_path(self, config):
        data_location = config['DCA1000Config']['captureConfig']['fileBasePath']
        if os.path.isabs(data_location):
            return config
        data_location = os.path.join(self.cwd, data_location)
        data_location = os.path.abspath(data_location)
        config['DCA1000Config']['captureConfig']['fileBasePath'] = data_location
        return config

    def log(self, msg):
        print('[DCA1000]', msg)

if __name__ == '__main__':
    cfg = {'samples_per_chirp': 1000, 'chirps_per_frame': 64, 'slope': 21000000000000.0, 'ADC_rate': 6000000.0, 'frame_time': 0.04, 'fps': 1600.0}
    DCA1000Handler('1642', cfg, None, None)
    while True:
        time.sleep(1)