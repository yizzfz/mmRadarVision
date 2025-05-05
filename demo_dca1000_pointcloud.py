"""
This demo reads raw IQ data + point cloud from a single radar. 
It requires the OOB firmware! 
"""
import os
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'
import multiprocessing
from radar_handler import Radar
from visualizer import Visualizer_Raw_Pointcloud
from config import *
from logger import Logger
from dca1000 import DCA1000Handler
from util import parse_radarcfg
import signal
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import time
matplotlib.use('TkAgg')
np.set_printoptions(precision=4, suppress=True)

radar_to_use = 0            # select which radar(s) to use, int or list(int)
dataformat = 'raw'          # 'raw' for raw data and 'fft' for range-FFT data
data_saving_location = 'C:/mmwave-log'

def vis_raw_thread(dca1000_queue, pointcloud_queue, runflag, config):
    """Visulization thread that hosts the Visualizer module and the peripherals"""
    print(f'[main] Visualizer PID {multiprocessing.current_process().pid}')
    logger = Logger('demo', path=data_saving_location)
    # initialize the visuzlizer for raw data processing 
    vis = Visualizer_Raw_Pointcloud(config, dca1000_queue, pointcloud_queue, runflag, polar=None, logger=logger, dataformat=dataformat)
    vis.run()
 
def radar_thread(queue, runflag, radar):
    """Radar thread to operate one radar"""
    # e.g. radar = ('1443A', 'COM6', 'COM7', './iwr1443/cfg/zoneA.cfg')
    name, cfg_port, data_port, cfg_file, _, _ = radar
    print(f'[main] Radar {name} PID {multiprocessing.current_process().pid}')
    radar = Radar(name, cfg_port, data_port, runflag, studio_cli_image=False, debug=False)
    success = radar.connect(cfg_file)
    if not success:
        raise ValueError(f'Radar {name} Connection Failed')
    radar.run(queue)

def dca1000_thread(queue, runflag, name, config):
    print(f'[main] DCA1000 PID {multiprocessing.current_process().pid}')
    dca1000 = DCA1000Handler(name, config, runflag, queue, data_format=dataformat)
    dca1000.run()

def main():
    # define which radar(s) to use
    radar_cfg = RADAR_CFG[radar_to_use]
    # parse the radar configuration file into a dict
    config = parse_radarcfg(radar_cfg[3])
    # define a shared variable to control the running status of the system.
    runflag = multiprocessing.Value('i', 1)
    # each radar needs a data queue to comminucate with the visualizer
    radar_queue = multiprocessing.Queue()
    # the DCA1000EVM module needs a data queue to comminucate with the visualizer
    dca1000_queue = multiprocessing.Queue()

    # to make the system exit on ctrl-c
    def signal_handler(*args):
        print('Exiting')
        runflag.value = 0
    signal.signal(signal.SIGINT, signal_handler)

    # define one visualizer thread
    threads = []
    t1 = multiprocessing.Process(target=vis_raw_thread, args=(dca1000_queue, radar_queue, runflag, config))
    threads.append(t1)

    # define one DCA1000EVM thread
    t2 = multiprocessing.Process(target=dca1000_thread, args=(dca1000_queue, runflag, radar_cfg[0], config))
    t2.start()      # start before radar
    time.sleep(1)

    # define one radar thread
    t3 = multiprocessing.Process(target=radar_thread, args=(radar_queue, runflag, radar_cfg))
    threads.append(t3)

    # start the system
    for t in threads:
        t.start()

    for t in threads:
        t.join()
    t2.join()


if __name__ == '__main__':
    main()
