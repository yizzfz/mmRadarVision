import multiprocessing
import numpy as np
from radar_handler import Radar
from visualizer import Visualizer_3D, Visualizer_Single, Visualizer_Multi, Visualizer_Cam_Single, Visualizer_Background
from frame_manager import Frame_Manager_Base, Frame_Manager_Cluster, Frame_Manager_Foreground
from detector import Detector_Human

radar_height = 1.5
d_hor = 1.5
d_ver = 0.9


radar_to_use = [0]
radars_all = [
    ('1443A', 'COM6', 'COM7', './iwr1443/cfg/zoneA.cfg'),       # 0
    ('1443B', 'COM17', 'COM16', './iwr1443/cfg/zoneB.cfg'),     # 1
    ('1642', 'COM12', 'COM13', './iwr1642/cfg/profile.cfg'),    # 2
    ('6843A', 'COM19', 'COM18', './iwr6843/cfg/profileA.cfg'),  # 3
    ('6843B', 'COM17', 'COM16', './iwr6843/cfg/profileB.cfg')   # 4
    ]


def vis_thread(num_radar, queues, runflag):
    if num_radar == 1:
        train_frame = 1000
        fm0 = Frame_Manager_Base(max_length=10, ylim=[0, 3], zlim=[-1, 1])
        fm1 = Frame_Manager_Foreground(max_length=1, train_frame=train_frame)
        fm2 = Frame_Manager_Cluster(max_length=1, min_points=5)

        # vis = Visualizer_Multi(queues, [fm1, fm2], n_row=1, n_col=2)
        # vis = Visualizer_3D(queues, [fm0, fm2])
        # detector = Detector_Human()
        vis = Visualizer_Cam_Single(
            queues, [fm0, fm1, fm2], detector=None, detector_start=0, save=False)
        # vis = Visualizer_Multi(queues, [fm0, fm1, fm2], n_row=1, n_col=3)
        # vis = Visualizer_Background(queues, [], save=True)
        vis.run(runflag)
 

def radar_thread(queue, runflag, radar):
    name, cfg_port, data_port, cfg_file = radar
    radar = Radar(name, cfg_port, data_port)
    radar.start(cfg_file, queue, runflag)
    

def main():
    radars = [radars_all[i] for i in radar_to_use]
    runflag = multiprocessing.Value('i', 1)
    num_radar = len(radar_to_use)
    queues = []

    for _ in range(num_radar):
        q = multiprocessing.Queue()
        queues.append(q)


    t0 = multiprocessing.Process(target=vis_thread, args=(
        num_radar, queues, runflag, ))
    threads = [t0]


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
