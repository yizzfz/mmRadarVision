import cv2
import numpy as np
import pickle
import serial
import time
import multiprocessing


class Motor():
    def __init__(self, port, pos):
        self.portno = port
        self.tbidx = 0
        with open('./motor/ruler.pkl', 'rb') as f:
            self.table = pickle.load(f)
        self.forward = True
        self.step = 0
        self.basetime = 0
        self.pos = pos

    def start(self, runflag):
        self.run = runflag
        try:
            self.port = serial.Serial(self.portno, 9600, timeout=10)
        except serial.serialutil.SerialException:
            self.log('Failed opening serial port for motor')
            self.run.value = 0
            return -1

        assert self.port.is_open

        time.sleep(0.5)

        data_line = self.port.readline().decode()
        self.log(data_line)
        self.port.write('125\n'.encode())
        self.log('connected')

        self.tbidx = 0

        data_line = ''
        while 'ready' not in data_line:
            data_line = self.port.readline().decode()
            self.log(data_line)

        time.sleep(5)
        self.port.write('a\n'.encode())
        self.basetime = time.time()

        while self.run is True or self.run.value == 1:
            while time.time()-self.basetime < 5:
                time.sleep(0.01)
                self.update_pos()
            data_line = self.port.readline()
            self.port.write('a\n'.encode())
            self.update_pos(reset=True)

    def log(self, text):
        print('[motor]', text)

    def update_pos(self, reset=False):
        if reset:
            self.basetime = time.time()
            self.tbidx = len(self.table)-1 if self.forward else 0
            self.forward = not self.forward

        self.tbidx = int((time.time()-self.basetime)*60)
        if not self.forward:
            self.tbidx = -self.tbidx
        self.step += 1
        if self.pos is not None:
            self.pos.value = self.table[self.tbidx]

if __name__ == '__main__':
    m = Motor('COM4', None)
    m.start(True)
