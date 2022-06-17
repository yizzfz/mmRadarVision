import multiprocessing
from radar_handler import Radar
from visualizer import Visualizer_Base, Visualizer_AE, Visualizer_TwoR_Tracker, Visualizer_TwoR_Vertical
from visualizer import Visualizer_Single_3D, Visualizer_Cam, Visualizer_NN, Visualizer_Range_Profile, Visualizer_Single_FM_Stages, Visualizer_TwoR
from frame_manager import Frame_Manager_Base, Frame_Manager_Cluster, Frame_Manager_Foreground
from config import *
from logger import Logger
from camera import Camera_Base
from heart_sensor import H10
import matplotlib
import numpy as np
matplotlib.use('TkAgg')
np.set_printoptions(precision=4, suppress=True)



radar_to_use = [2]
camera = None
heart_sensor = 'c1:02:a3:7c:57:42'
# heart_sensor = None

def vis_thread(num_radar, queues, runflag, cam=None, heart_sensor=None):
    if cam is not None:
        cam = Camera_Base(cam)
    if heart_sensor is not None:
        heart_sensor = H10(heart_sensor, task='hr', data_only=True)
    logger = Logger('tmp', path='d:/mmwave-log')
    logger = None

    if num_radar == 1:
        fm0 = Frame_Manager_Base(max_length=5, xlim=[-1, 1], ylim=[0.2, 3], zlim=[-1, 1])
        fm1 = Frame_Manager_Foreground(max_length=1, train_frame=1000)
        fm2 = Frame_Manager_Cluster(max_length=1, min_points=5)
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
    radar = Radar(name, cfg_port, data_port, runflag)
    success = radar.connect(cfg_file)
    if not success:
        raise ValueError(f'Radar {name} Connection Failed')
    radar.run(queue)

def main():
    radars = [RADAR_CFG[i] for i in radar_to_use]
    runflag = multiprocessing.Value('i', 1)
    num_radar = len(radar_to_use)
    queues = []
    threads = []

    for _ in range(num_radar):
        q = multiprocessing.Queue()
        queues.append(q)

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
