import os
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'
import multiprocessing
from radar_handler import Radar
from visualizer import Visualizer_AE, Visualizer_TwoR, Visualizer_TwoR_Vertical
from frame_manager import Frame_Manager_Base, Frame_Manager_Cluster, Frame_Manager_Foreground
from config import *
from logger import Logger
from camera import Camera_Base
from heart_sensor import Polar
import matplotlib
import numpy as np
import signal

matplotlib.use('TkAgg')
np.set_printoptions(precision=4, suppress=True)

radar_to_use = [2, 3]   # select which radar(s) to use, int or list(int)
camera = None           # select which camera to use, int or None
heart_sensor = None     # select which heart rate sensor to use using the MAC addr, str

def vis_thread(num_radar, queues, runflag, cam=None, heart_sensor=None):
    """Visulization thread that hosts the Visualizer module and the peripherals"""
    print(f'[main] Visualizer PID {multiprocessing.current_process().pid}')
    if cam is not None:                             # configure the camera
        cam = Camera_Base(cam)
    if heart_sensor is not None:                    # configure the heart rate sensor
        heart_sensor = Polar(heart_sensor, task='hr', data_only=True)
    logger = Logger('tmp', path='D:/mmwave-log')    # configure the logger

    if num_radar == 1:
        # one frame manager to stack points in neighbour frames and filter out irrelvant points
        fm0 = Frame_Manager_Base(max_length=25, xlim=[-1, 1], ylim=[0.2, 3], zlim=[-1, 1])
        # default visualizer for one radar is Visualizer_AE
        vis = Visualizer_AE(queues, [[fm0]], logger=logger, xlim=[-1, 1], ylim=[0.2, 3], zlim=[0, 2], height=[1.3], cam=cam, heart_sensor=heart_sensor)
        # start the visualizer
        vis.run(runflag)
    elif num_radar == 2:
        # one frame manager for each radar
        fm01 = Frame_Manager_Base(max_length=12, xlim=[-1.5, 1.5], ylim=[0.2, 5], zlim=[-1, 1])
        fm02 = Frame_Manager_Base(max_length=12, xlim=[-1.5, 1.5], ylim=[0.2, 5], zlim=[-1, 1])
        # use a visualizer starts with `Visualizer_TwoR` for two radars
        vis = Visualizer_TwoR_Vertical(queues, [[fm01], [fm02]], xlim=[-1.5, 1.5], ylim=[0.2, 5], zlim=[0, 2],
                                       height=[radar_height1, radar_height2], cam=cam, logger=logger, heart_sensor=heart_sensor)
        # start the visualizer
        vis.run(runflag)
 
def radar_thread(queue, runflag, radar):
    """Radar thread to operate one radar"""
    # e.g. radar = ('1443A', 'COM6', 'COM7', './iwr1443/cfg/zoneA.cfg')
    name, cfg_port, data_port, cfg_file = radar
    print(f'[main] Radar {name} PID {multiprocessing.current_process().pid}')

    radar = Radar(name, cfg_port, data_port, runflag, outformat='pvs')
    success = radar.connect(cfg_file)
    if not success:
        raise ValueError(f'Radar {name} Connection Failed')
    radar.run(queue)

def main():
    # define which radar(s) to use
    radars = [RADAR_CFG[i] for i in radar_to_use]
    # define a shared variable to control the running status of the system.
    runflag = multiprocessing.Value('i', 1)
    num_radar = len(radar_to_use)
    # each radar needs a data queue to comminucate with the visualizer
    queues = []
    for _ in range(num_radar):
        q = multiprocessing.Queue()
        queues.append(q)

    # to make the system exit on ctrl-c
    def signal_handler(*args):
        print('Exiting')
        runflag.value = 0
    signal.signal(signal.SIGINT, signal_handler)

    # define one visualizer thread
    threads = []
    t0 = multiprocessing.Process(target=vis_thread, args=(num_radar, queues, runflag, camera, heart_sensor))
    threads.append(t0)

    # define one radar thread for each radar
    for i in range(num_radar):
        t = multiprocessing.Process(target=radar_thread, args=(queues[i], runflag, radars[i]))
        threads.append(t)

    # start the system
    for t in threads:
        t.start()

    for t in threads:
        t.join()

if __name__ == '__main__':
    main()
