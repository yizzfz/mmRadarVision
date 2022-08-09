import pickle
import datetime
import os

class Logger():
    """Logger module for data recording."""
    def __init__(self, name, path='./log'):
        """
        Parameters:
            name: name of the record result file.
            path: path of the record result file.
        """
        timestamp = datetime.datetime.now().strftime('%m%d-%H%M')
        if not os.path.exists(path):
            os.mkdir(path)
        self.logfile = os.path.abspath(os.path.join(path, f'{timestamp}-{name}.pkl'))
        self.D = {}
        self.header = {}
        self.t0 = None
        self.t1 = None

    def update(self, data, datatype='misc'):
        """Store one frame of data of certain type"""
        if self.t0 is None:
            self.t0 = datetime.datetime.now()
        self.t1 = datetime.datetime.now()
        if not datatype in self.D:
            self.D[datatype] = []
        self.D[datatype].append(data)

    def set_header(self, header:dict):
        """Append a header to the result file"""
        self.header = dict(self.header, **header)

    def save(self):
        """Save the recorded data to disk"""
        data_length = 0
        for i in self.D:
            length = len(self.D[i])
            data_length = max(data_length, length)
        if data_length == 0:
            self.log('No data recorded')
            return

        time_elapsed = (self.t1-self.t0).total_seconds()
        self.header['start_time'] = self.t0.strftime('%m.%d-%H:%M:%S')
        self.header['stop_time'] = self.t1.strftime('%m.%d-%H:%M:%S')
        self.header['time_elapsed'] = time_elapsed
        self.header['data_types'] = list(self.D.keys())
        self.header['num_data'] = data_length
        self.log(f'Saving {data_length} data ({time_elapsed:.1f} seconds) to {self.logfile}')
        with open(self.logfile, 'wb') as f:
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
