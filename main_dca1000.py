import os
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'
import multiprocessing
from radar_handler import Radar
from visualizer import Visualizer_Base, Visualizer_Raw, Visualizer_TwoR, Visualizer_HeartRate
from visualizer import Visualizer_HeartRate_Basic, Visualizer_HeartRate_NN, Visualizer_HeartRate_Alpha
from frame_manager import Frame_Manager_Base, Frame_Manager_Cluster, Frame_Manager_Foreground
from config import *
from logger import Logger
from camera import Camera_Base
from heart_sensor import Polar
from dca1000 import DCA1000Handler
from util import parse_radarcfg
import signal
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import time
matplotlib.use('TkAgg')
np.set_printoptions(precision=4, suppress=True)

radar_to_use = 5
camera = None
polar_addr = POLAR['H10']
data_format = 'fft'
polar = None
logger = None

def vis_raw_thread(queue, runflag, config):
    print(f'[main] Visualizer PID {multiprocessing.current_process().pid}')
    polar = Polar(polar_addr, task='hr', data_only=True)
    logger = Logger('HR', path='D:/mmwave-log/hr')

    if data_format == 'raw':
        vis = Visualizer_Raw(config, queue, runflag, polar=polar, logger=logger)
    else:
        vis = Visualizer_Raw(config, queue, runflag, polar=polar, logger=logger)
        # vis = Visualizer_HeartRate_Alpha(config, queue, runflag, polar=polar, logger=logger)
        # vis = Visualizer_HeartRate_NN(config, queue, runflag, polar=polar, logger=logger)
    vis.run()
 
def radar_thread(queue, runflag, radar):
    name, cfg_port, data_port, cfg_file = radar
    print(f'[main] Radar {name} PID {multiprocessing.current_process().pid}')
    # e.g. radar = ('1443A', 'COM6', 'COM7', './iwr1443/cfg/zoneA.cfg')
    radar = Radar(name, cfg_port, data_port, runflag, studio_cli_image=True)
    success = radar.connect(cfg_file)
    if not success:
        raise ValueError(f'Radar {name} Connection Failed')
    radar.run(queue)

def dca1000_thread(queue, runflag, name, config):
    print(f'[main] DCA1000 PID {multiprocessing.current_process().pid}')
    dca1000 = DCA1000Handler(name, config, runflag, queue, data_format=data_format)
    dca1000.run()

def main():
    radar_cfg = RADAR_CFG[radar_to_use]
    config = parse_radarcfg(radar_cfg[3])
    runflag = multiprocessing.Value('i', 1)
    radar_queue = multiprocessing.Queue()
    dca1000_queue = multiprocessing.Queue()

    def signal_handler(*args):
        print('Exiting')
        runflag.value = 0
    signal.signal(signal.SIGINT, signal_handler)

    threads = []
    t1 = multiprocessing.Process(target=vis_raw_thread, args=(dca1000_queue, runflag, config))
    threads.append(t1)

    t2 = multiprocessing.Process(target=dca1000_thread, args=(dca1000_queue, runflag, radar_cfg[0], config))
    t2.start()      # start before radar
    time.sleep(1)

    t3 = multiprocessing.Process(target=radar_thread, args=(radar_queue, runflag, radar_cfg))
    threads.append(t3)


    for t in threads:
        t.start()

    for t in threads:
        t.join()
    t2.join()


if __name__ == '__main__':
    main()
