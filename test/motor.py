import serial
import time
import pickle
import struct
import numpy as np


class Motor():
    def __init__(self, port):
        self.portno = port
        self.pos = 0

    def start(self, runflag):
        self.run = runflag
        try:
            port = serial.Serial(self.portno, 9600, timeout=1)
        except serial.serialutil.SerialException:
            self.log('Failed opening serial port for motor')
            self.run.value = 0
            return -1

        assert port.is_open

        time.sleep(0.5)
        data_line = port.readline()
        print(data_line.decode())
        port.write('125\n'.encode())

        self.log('connected')
        
        while self.run is True or self.run.value == 1:
            time.sleep(5)
            data_line = port.readline()
            port.write('a\n'.encode())
            print(data_line.decode())

    def log(self, text):
        print(text)

m = Motor('COM4')
m.start(True)

