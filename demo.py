import os
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'
import multiprocessing
from radar_handler import Radar
from visualizer import Visualizer_AE, Visualizer_TwoR
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

radar_to_use = [0]
camera = None
heart_sensor = None

def vis_thread(num_radar, queues, runflag, cam=None, heart_sensor=None):
    print(f'[main] Visualizer PID {multiprocessing.current_process().pid}')
    if cam is not None:
        cam = Camera_Base(cam)
    if heart_sensor is not None:
        heart_sensor = Polar(heart_sensor, task='hr', data_only=True)
    logger = Logger('tmp', path='d:/mmwave-log')
    # logger = None

    if num_radar == 1:
        fm0 = Frame_Manager_Base(max_length=5, xlim=[-1, 1], ylim=[0.2, 3], zlim=[-1, 1])
        vis = Visualizer_AE(queues, [[fm0]], logger=logger, xlim=[-1, 1], ylim=[0.2, 3], zlim=[0, 2], height=[1.2], cam=cam, heart_sensor=heart_sensor)
        vis.run(runflag)
    elif num_radar == 2:
        fm01 = Frame_Manager_Base(max_length=12, xlim=[-0.75, 0.75], ylim=[0.2, 5], zlim=[-1, 1])
        fm02 = Frame_Manager_Base(max_length=12, xlim=[-0.75, 0.75], ylim=[0.2, 5], zlim=[-1, 1])
        fm11 = Frame_Manager_Cluster(max_length=1, min_points=8)
        fm12 = Frame_Manager_Cluster(max_length=1, min_points=8)
        vis = Visualizer_TwoR(queues, [[fm01, fm11], [fm02, fm12]], xlim=[-1.5, 1.5], ylim=[0, 1.5], cam=cam, logger=logger, heart_sensor=heart_sensor)
        vis.run(runflag)
 
def radar_thread(queue, runflag, radar):
    name, cfg_port, data_port, cfg_file = radar
    # e.g. radar = ('1443A', 'COM6', 'COM7', './iwr1443/cfg/zoneA.cfg')
    print(f'[main] Radar {name} PID {multiprocessing.current_process().pid}')

    radar = Radar(name, cfg_port, data_port, runflag, outformat='pvs')
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
        print('Exiting')
        runflag.value = 0
    signal.signal(signal.SIGINT, signal_handler)

    threads = []
    t0 = multiprocessing.Process(target=vis_thread, args=(num_radar, queues, runflag, camera, heart_sensor))
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
