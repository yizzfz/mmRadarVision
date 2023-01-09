import pickle
import datetime
import os
import time
from collections import defaultdict


class Logger():
    """Logger module for data recording."""
    def __init__(self, name, path='./log', save_interval=30):
        """
        Parameters:
            name: name of the record result file.
            path: path of the record result file.
        """
        timestamp = datetime.datetime.now().strftime('%m%d-%H%M')
        if not os.path.exists(path):
            os.mkdir(path)
        self.logfile = os.path.abspath(os.path.join(path, f'{timestamp}-{name}.pkl'))
        self.logfile_partial = os.path.abspath(os.path.join(path, f'{timestamp}-{name}'))
        self.D = {}
        self.header = {}
        self.t0 = None
        self.t1 = None

        self.postfix = 0
        self.save_interval = save_interval
        self.starttime = datetime.datetime.now()

    def update(self, data, datatype='misc'):
        """Store one frame of data of certain type"""
        if self.t0 is None:
            self.t0 = datetime.datetime.now()
        self.t1 = datetime.datetime.now()
        if not datatype in self.D:
            self.D[datatype] = []
        self.D[datatype].append(data)
        if (datetime.datetime.now() - self.starttime).total_seconds() > self.save_interval:
            self.save_at_interval()
            self.starttime = datetime.datetime.now()
    
    def save_at_interval(self):
        self.save(custom_logfile=f'{self.logfile_partial}.{self.postfix}.pkl')
        self.D = defaultdict(list)
        self.postfix += 1

    def set_header(self, header:dict):
        """Append a header to the result file"""
        self.header = dict(self.header, **header)

    def save(self, custom_logfile=None):
        """Save the recorded data to disk"""
        data_length = 0
        for i in self.D:
            length = len(self.D[i])
            data_length = max(data_length, length)
        if data_length == 0:
            self.log('No data recorded')
            return

        logfile = self.logfile if custom_logfile is None else custom_logfile
        time_elapsed = (self.t1-self.t0).total_seconds()
        self.header['start_time'] = self.t0.strftime('%m.%d-%H:%M:%S')
        self.header['stop_time'] = self.t1.strftime('%m.%d-%H:%M:%S')
        self.header['time_elapsed'] = time_elapsed
        self.header['data_types'] = list(self.D.keys())
        self.header['num_data'] = data_length
        self.log(f'Saving {data_length} data ({time_elapsed:.1f} seconds) to {logfile}')
        with open(logfile, 'wb') as f:
            pickle.dump(self.header, f, protocol=pickle.HIGHEST_PROTOCOL)
            for i in self.D:
                pickle.dump(self.D[i], f, protocol=pickle.HIGHEST_PROTOCOL)
        self.log(f'Saved')

    def log(self, msg):
        print('[Data logger]', msg)


if __name__ == '__main__':
    log = Logger('tmp')
    log.update('txt')
    log.set_header({'test header':'nothing'})
    log.save()
