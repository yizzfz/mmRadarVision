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


dependency = ['DCA1000EVM_CLI_Control.exe', 'DCA1000EVM_CLI_Record.exe', 'RF_API.dll']

class DCA1000Handler:
    """Only designed with 1 RX"""
    def __init__(self, model, radarcfg, runflag, queue, send_rate=1, port=60203, data_format='fft', fft_multiplier=1):
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
        self.data_per_packet = 0
        self.bytes_per_packet = 0
        self.send_rate = send_rate
        self.fft_multiplier = fft_multiplier
        self.cfg_name = 'tmp.json'
        for f in dependency:
            assert os.path.exists(os.path.join(self.cwd, f)) and f"{f} is required to run DCA1000"
        self.parse_radarcfg(radarcfg)
        # self.data_epilog = None
        assert self.data_per_packet > 0 and "Failed to read radar cfg"

        jsonfile = 'dca1000.json'
        with open(os.path.join(self.cwd, jsonfile)) as f:
            dca1000config = json.load(f)
        dca1000config = self.convert_to_abs_path(dca1000config)
        basepath = dca1000config['DCA1000Config']['captureConfig']['fileBasePath']
        if not os.path.exists(basepath):
            os.mkdir(basepath)
        if '1642' in model or '6843' in model or '1843' in model:
            dca1000config['DCA1000Config']['lvdsMode'] = 2
        else:
            raise ValueError(f'radar not supported')
        dca1000config['DCA1000Config']['captureConfig']['filePrefix'] = model
        self.data_location = dca1000config['DCA1000Config']['captureConfig']['fileBasePath']
        self.data_prefix = dca1000config['DCA1000Config']['captureConfig']['filePrefix']
        self.packet_delay = int(dca1000config['DCA1000Config']['packetDelay_us'])
        self.allowed_bandwidth = 12000/(12+self.packet_delay) / 8
        # self.data_size = dca1000config['DCA1000Config']['captureConfig']['maxRecFileSize_MB'] * 1e6       # in bytes
        self.log(f'Packet delay set to {self.packet_delay} us, bandwidth {self.allowed_bandwidth:.2f} MB/s')
        self.log(f'Expecting {self.bytes_per_packet * self.send_rate:,.0f} bytes of data per seconds, data packet shape {self.data_sp_shape}')
        # self.log(f'Each file should contain {self.data_size/self.bytes_per_second:.2f} seconds of data')

        self.FP = FFTProcessor(radarcfg, multiplier=fft_multiplier, max_d=5)
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
        while len(data) < self.bytes_per_packet:
            packet = socket.recv(self.bytes_per_packet - len(data))
            if not packet:
                raise ValueError('No data received')
            data.extend(packet)
        return data

    def adc_format(self, adcData):
        if '1642' in self.model or '6843' in self.model or '1843' in self.model:
            '''
            Data in:    n_frame, n_chirp, n_rx, n_samples/2, IQ, 2
            Data out:   n_frame, n_chirp, n_samples, n_rx
            '''
            adcData = adcData.reshape((-1, 4))
            adcData = adcData[:, 0:2] + 1j*adcData[:, 2:4]
        elif '1443' in self.model:
            '''
            Data in:    n_frame, n_chirp, n_sample, IQ, n_rx
            Data out:   n_frame, n_chirp, n_sample, n_rx
            '''
            adcData = adcData.reshape((-1, 2))
            adcData = adcData[:, 0] + 1j*adcData[:, 1]
        else:
            raise ValueError('Radar model not supported')
        adcData = adcData.reshape(self.data_sp_shape)
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
                            datashape = fftData.shape if self.data_format == 'fft' else adcData.shape
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

        self.frame_per_packet = int(self.frames_per_second/self.send_rate)
        self.data_per_packet = self.samples_per_chirp * self.chirps_per_frame * self.frame_per_packet
        self.bytes_per_packet = self.data_per_packet * 4
        self.data_sp_shape = 1, self.frame_per_packet * self.chirps_per_frame, self.samples_per_chirp

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

class DCA1000Handler_MIMO(DCA1000Handler):
    def __init__(self, model, radarcfg, runflag, queue, send_rate=1, port=60203, data_format='fft', fft_multiplier=1):
        super().__init__(model, radarcfg, runflag, queue, send_rate, port, data_format, fft_multiplier)

    def parse_radarcfg(self, config):
        super().parse_radarcfg(config)
        self.n_rx = config['n_rx']
        self.chirploops_per_frame = config['chirploops_per_frame']
        self.chirps_per_loop = config['chirps_per_loop']
        self.data_per_packet = self.samples_per_chirp * self.chirps_per_frame * self.frame_per_packet * self.n_rx
        self.bytes_per_packet = self.data_per_packet * 4
        self.data_sp_shape = self.n_rx*self.chirps_per_loop, self.frame_per_packet * self.chirploops_per_frame, self.samples_per_chirp

    def adc_format(self, adcData):
        if '1642' in self.model or '6843' in self.model or '1843' in self.model:
            '''
            Data in:    (n_frame, n_chirp), n_rx, n_samples/2, IQ, 2
            Data out:   (n_frame, n_chirp), n_samples, n_rx
            '''
            adcData = adcData.reshape((-1, 2, 2))
            adcData = adcData[:, 0] + 1j*adcData[:, 1]
            adcData = adcData.reshape(-1, self.n_rx, self.samples_per_chirp)
            adcData = np.transpose(adcData, (0, 2, 1))
        elif '1443' in self.model:
            '''
            Data in:    (n_frame, n_chirp, n_sample), IQ, n_rx
            Data out:   (n_frame, n_chirp, n_sample), n_rx
            '''
            adcData = adcData.reshape((-1, 2, 4))
            adcData = adcData[:, 0, :] + 1j*adcData[:, 1, :]
            adcData = adcData.reshape(-1, 4)[:, :self.n_rx]
        else:
            raise ValueError('Radar model not supported')
        adcData = self.TDM_shape(adcData)
        assert adcData.shape == self.data_sp_shape
        return adcData

    def TDM_shape(self, adcData):
        """ 
        In: n_frame, n_chirp (=n_chirploop*n_tx), n_sample, n_rx
        Out: n_frame * n_chirploop, n_sample, n_rx * n_tx
        """
        adcData = adcData.reshape((self.frame_per_packet * self.chirploops_per_frame, self.chirps_per_loop, self.samples_per_chirp, self.n_rx))
        adcData = np.transpose(adcData, (1, 3, 0, 2))
        adcData = adcData.reshape((-1, *adcData.shape[2:]))
        return adcData


class FFTProcessor:
    def __init__(self, config, multiplier=4, max_d=1.5):
        """config must include: fps, samples_per_chirp, ADC_rate, slope"""
        self.config = config
        self.multiplier = multiplier
        self.max_d = max_d
        n_samples = self.config['samples_per_chirp']
        ADC_rate = self.config['ADC_rate']
        slope = self.config['slope']
        self.n_fft = n_samples*multiplier
        fft_freq = np.fft.fftfreq(self.n_fft, d=1.0/ADC_rate)
        fft_freq_d = fft_freq*3e8/2/slope
        self.max_freq_i = np.argmax(fft_freq_d>self.max_d)
        self.fft_freq = fft_freq[:self.max_freq_i]
        self.win = np.hanning(n_samples)

    def compute_FFTs(self, data, split=True):
        """
        Parameters:
            data: shape (n_rx, n_chirp, n_sample).
            split: return seperate mag and phase or raw FFT output
        """
        data = data * self.win
        fft_out = np.fft.fft(data, self.n_fft)[:, :, :self.max_freq_i]
        if split:
            fft_mags = np.abs(fft_out)
            fft_phases = np.angle(fft_out)/np.pi
            return np.stack((fft_mags, fft_phases))
        return fft_out

    def get_fft_freq(self):
        return self.fft_freq

if __name__ == '__main__':
    cfg = {'samples_per_chirp': 1000, 'chirps_per_frame': 64, 'slope': 21000000000000.0, 'ADC_rate': 6000000.0, 'frame_time': 0.04, 'fps': 1600.0}
    DCA1000Handler('1642', cfg, None, None)
    while True:
        time.sleep(1)