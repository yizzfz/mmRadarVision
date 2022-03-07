import multiprocessing
import numpy as np
from radar_handler import Radar
from visualizer import Visualizer_Single, Visualizer_Single_Motor, Visualizer_Single_P
from visualizer import Visualizer_3D, Visualizer_Multi
from visualizer import Visualizer_Cam_Single, Visualizer_Cam_Data, Visualizer_Background
from visualizer import Visualizer_NN
from visualizer_two import Visualizer_Base_2R, Visualizer_Cam_2R, Visualizer_Cam_2R_eval, Visualizer_2R_Tracker
from visualizer_two import Visualizer_Base_2P, Visualizer_Base_2P_9S, Visualizer_2P_Pose
from visualizer_two import Visualizer_2P_Pose_NN, Visualizer_2P_Pose_vs6
from visualizer import Visualizer_Range_Profile
from frame_manager import Frame_Manager_Base, Frame_Manager_Cluster, Frame_Manager_Foreground, Frame_Manager_Motor
from config import *
from logger import Logger
# from network import *
# from motor import Motor

from camera import Camera_Simple
import matplotlib
matplotlib.use('TkAgg')

radar_to_use = []
radar_to_use = [0, 3]
use_motor = 0
interference = False


def vis_thread(num_radar, queues, runflag, motorpos):
    logger = None
    cam = None
    # logger = Logger('pose_rawdata_cam', path='d:/mmwave-log')
    # cam = Camera_360(cam=2)
    # nn = Pose_Net(load='./network/pose_net_ckpts/09192023')

    if interference:
        vis = Visualizer_Range_Profile(queues, save=True)
        vis.run(runflag)
        return

    if num_radar == 1:
        train_frame = 1000
        fm0 = Frame_Manager_Base(max_length=5, xlim=[-1, 1], ylim=[0.2, 3], zlim=[-1, 1])
        # fm1 = Frame_Manager_Foreground(max_length=1, train_frame=train_frame)
        fm2 = Frame_Manager_Cluster(max_length=1, min_points=5)

        # nn = Simple_Net()
        # nn.load_checkpoint('07291026')

        # vis = Visualizer_NN(queues, [fm0, fm2], model=nn)
        # vis = Visualizer_Multi(queues, [fm1, fm2], n_row=1, n_col=2)
        # vis = Visualizer_3D(queues, [fm0, fm2])
        # detector = Detector_Human()
        # vis = Visualizer_Cam_Data(
        #     queues, [fm0], detector=Detector_Human(min_prob=90), detector_start=0, save=True)
        # vis = Visualizer_Multi(queues, [fm0, fm1, fm2], n_row=1, n_col=3)
        # vis = Visualizer_Background(queues, [], save=True)
        if use_motor:
            assert(motorpos is not None)
            fm_motor = Frame_Manager_Motor(motorpos, max_length=1)
            vis = Visualizer_Single_P(queues, [fm_motor, fm0], ylim=[0, 10])
            vis.run(runflag)
        vis = Visualizer_Single_P(queues, [fm0], logger=None, xlim=[-1, 1], ylim=[0.2, 3], zlim=[0, 2], height=1.2, cam=cam)
        vis.run(runflag)
        # vis = Visualizer_Base_2R(queues, [])

    elif num_radar == 2:
        fm01 = Frame_Manager_Base(max_length=12, xlim=[-0.75, 0.75], ylim=[0.2, 5], zlim=[-1, 1])
        fm02 = Frame_Manager_Base(
            max_length=12, xlim=[-0.75, 0.75], ylim=[0.2, 5], zlim=[-1, 1])

        fm11 = Frame_Manager_Cluster(max_length=1, min_points=8)
        fm12 = Frame_Manager_Cluster(max_length=1, min_points=8)

        vis = Visualizer_Base_2R(queues, [fm01, fm11], [fm02, fm12], plot_mode='simple')
        # vis = Visualizer_2R_Tracker(queues, [fm01, fm11], [fm02, fm12])
        # vis = Visualizer_2P_Pose(queues, [fm01], [fm02],
        #                          height1=height_1, height2=height_2,
        #                          zlim=[0, 2], logger=logger, cam=cam)
        # vis = Visualizer_2P_Pose_vs6(queues, nn, fm1=[fm01, fm11], fm2=[fm02, fm12],
        #                             height1=height_1, height2=height_2,
        #                             zlim=[0, 2])

        # vis = Visualizer_Base_2R(queues, [fm01, fm11], [fm02, fm12], logger=logger)
        # vis = Visualizer_Cam_2R(queues, [fm01, fm11], [fm02, fm12], detector=Detector_Human(), save=False)
        # vis = Visualizer_Cam_2R_eval(queues, [fm01, fm11], [fm02, fm12], detector=Detector_Human(), save=True)
        vis.run(runflag)
    if cam:
        cam.stop()
 

def radar_thread(queue, runflag, radar):
    name, cfg_port, data_port, cfg_file = radar
    # e.g. radar = ('1443A', 'COM6', 'COM7', './iwr1443/cfg/zoneA.cfg')
    radar = Radar(name, cfg_port, data_port, runflag)
    success = radar.connect(cfg_file)
    if not success:
        return
    # if name == '1443A':
    #     radar.run(queue)
    # else:
    #     pass
    radar.run(queue)
    
def motor_thread(runflag, pos):
    motor = Motor(motor_port, pos)
    motor.start(runflag)


def main():
    assert(not use_motor or (use_motor and len(radar_to_use)==1))
    radars = [radar_ports[i] for i in radar_to_use]
    runflag = multiprocessing.Value('i', 1)
    num_radar = len(radar_to_use)
    queues = []
    threads = []
    motorpos = None

    for _ in range(num_radar):
        q = multiprocessing.Queue()
        queues.append(q)

    if use_motor:
        motorpos = multiprocessing.Value('f', 0.0)
        tm = multiprocessing.Process(
            target=motor_thread, args=(runflag, motorpos))
        threads.append(tm)

    t0 = multiprocessing.Process(target=vis_thread, args=(
        num_radar, queues, runflag, motorpos))
    threads.append(t0)


    for i in range(num_radar):
        t = multiprocessing.Process(target=radar_thread, args=(
            queues[i], runflag, radars[i]))
        threads.append(t)

    for t in threads:
        t.start()

    for t in threads:
        t.join()


if __name__ == '__main__':
    main()
