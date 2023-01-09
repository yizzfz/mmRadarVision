import os

from visualizer.vis_2r import Visualizer_TwoR_2D, Visualizer_TwoR_Tracker
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'
import multiprocessing
from radar_handler import Radar
from visualizer import Visualizer_AE, Visualizer_TwoR, Visualizer_TwoR_Vertical, Visualizer_TwoR_MAGG
from frame_manager import Frame_Manager_Base, Frame_Manager_Cluster, Frame_Manager_Foreground, Frame_Manager_Clutter_Removal
from config import *
from logger import Logger
from camera import Camera_Base, Camera_Simple
from heart_sensor import Polar
import matplotlib
import numpy as np
import signal
import warnings
from util import nchannel_from_dataformat

matplotlib.use('TkAgg')
np.set_printoptions(precision=4, suppress=True)

radar_to_use = [2, 3]   # select which radar(s) to use, int or list(int)
camera = None           # select which camera to use, int or None
heart_sensor = None     # select which heart rate sensor to use using the MAC addr, str
FORMAT = 'pv'

def vis_thread(radars, queues, runflag, cam=None, heart_sensor=None):
    if cam is not None:
        cam = Camera_Base(cam)
    if heart_sensor is not None:
        heart_sensor = Polar(heart_sensor, task='hr', data_only=True)
    logger = Logger('tmp', path='d:/mmwave-log')
    logger = None
    num_radar = len(radars)
    data_channel = nchannel_from_dataformat(FORMAT)
    if num_radar == 1:
        fm0 = Frame_Manager_Base(max_length=5, xlim=[-1, 1], ylim=[0.2, 5], zlim=[-1, 1])
        fm1 = Frame_Manager_Foreground(max_length=1, train_frame=1000)
        fm2 = Frame_Manager_Cluster(max_length=1, min_points=5)
        vis = Visualizer_AE(queues, [[fm0]], logger=logger, xlim=[-1, 1], ylim=[0.2, 3], zlim=[0, 2], height=[1.5], 
                            cam=cam, heart_sensor=heart_sensor, data_channel=data_channel)
        vis.run(runflag)
    elif num_radar == 2:
        # fm01 = Frame_Manager_Base(max_length=12)
        # fm02 = Frame_Manager_Base(max_length=12)
        # fm11 = Frame_Manager_Cluster(min_points=5, distance=0.3)
        # fm12 = Frame_Manager_Cluster(min_points=5, distance=0.3)
        # fm21 = Frame_Manager_Clutter_Removal()
        # fm22 = Frame_Manager_Clutter_Removal()

        fm1 = [
            Frame_Manager_Base(max_length=12, zlim=[0, 2]),
            Frame_Manager_Clutter_Removal(maxspeed=1),
            Frame_Manager_Cluster(min_points=20, distance=0.3),
        ]
        fm2 = [
            Frame_Manager_Base(max_length=12, zlim=[0, 2]),
            Frame_Manager_Clutter_Removal(maxspeed=1),
            Frame_Manager_Cluster(min_points=20, distance=0.3),
        ]
        vis = Visualizer_TwoR_MAGG(queues, [fm1, fm2], radars=radars,
                              xlim=[-2, 2], ylim=[-0.5, 4], zlim=[0.2, 1.8], plot_mode='final',
                              cam=cam, logger=logger, heart_sensor=heart_sensor, data_channel=data_channel)
        vis.run(runflag)
 

def radar_thread(queue, runflag, radar):
    name, cfg_port, data_port, cfg_file, rotation, translation = radar
    # e.g. radar = ('1443A', 'COM6', 'COM7', './iwr1443/cfg/zoneA.cfg')
    radar = Radar(name, cfg_port, data_port, runflag, rotation, translation, outformat=FORMAT, exit_on_fail=False)
    success = radar.connect(cfg_file)
    if not success:
        raise ValueError(f'Radar {name} Connection Failed')
    radar.run(queue)

def main():
    radars = [RADAR_CFG[i] for i in radar_to_use]
    runflag = multiprocessing.Value('i', 1)
    num_radar = len(radar_to_use)
    queues = []

    for _ in range(num_radar):
        q = multiprocessing.Queue()
        queues.append(q)

    def signal_handler(*args):
        print('System exiting')
        runflag.value = 0
    signal.signal(signal.SIGINT, signal_handler)

    threads = []
    t0 = multiprocessing.Process(target=vis_thread, args=(radars, queues, runflag, camera, heart_sensor))
    threads.append(t0)

    for i in range(num_radar):
        t = multiprocessing.Process(target=radar_thread, args=(queues[i], runflag, radars[i]))
        threads.append(t)

    for t in threads:
        t.start()

    for t in threads:
        t.join()

if __name__ == '__main__':
    main()
