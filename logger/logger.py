import pickle
import datetime
import os

class Logger():
    def __init__(self, name, path='./log'):
        timestamp = datetime.datetime.now().strftime('%m%d-%H%M')
        self.logfile = f'{path}/{timestamp}-{name}.pkl'
        self.radar_data = []
        self.cam_data = []
        self.misc_data = []
        self.heart_data = []
        self.D = {
            'radar': self.radar_data,
            'cam': self.cam_data,
            'heart': self.heart_data,
            'misc':self.misc_data
        }
        self.cwd = os.getcwd()
        self.header = None

    def update(self, data, datatype='misc'):
        self.D[datatype].append(data)

    def set_header(self, header):
        self.header = header

    def save(self):
        data_length = 0
        for i in self.D:
            length = len(self.D[i])
            data_length = max(data_length, length)
            print(f'[Data Logger] {i}: {length}')
        if data_length == 0:
            print('No data recorded')
            return

        os.chdir(self.cwd)
        with open(self.logfile, 'wb') as f:
            if self.header:
                pickle.dump(self.header, f)
            for i in self.D:
                pickle.dump(self.D[i], f)
        print(f'{data_length} data saved to {self.logfile}')
