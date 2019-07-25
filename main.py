import multiprocessing
import numpy as np
from radar_handler import Radar
from visualizer import Visualizer_3D, Visualizer_Single, Visualizer_Multi
from visualizer import Visualizer_Cam_Single, Visualizer_Cam_Data, Visualizer_Background
from visualizer import Visualizer_NN
from visualizer_two import Visualizer_Base_2R
from frame_manager import Frame_Manager_Base, Frame_Manager_Cluster, Frame_Manager_Foreground
# from detector import Detector_Human
from config import radar_ports
from network import Simple_Net



radar_to_use = [1, 0]



def vis_thread(num_radar, queues, runflag):
    if num_radar == 1:
        train_frame = 1000
        fm0 = Frame_Manager_Base(max_length=6, ylim=[0, 3], zlim=[-1, 1])
        fm1 = Frame_Manager_Foreground(max_length=1, train_frame=train_frame)
        fm2 = Frame_Manager_Cluster(max_length=1, min_points=5)

        # nn = Simple_Net()
        # nn.load_checkpoint('07231520')

        # vis = Visualizer_NN(queues, [fm0], model=nn)
        # vis = Visualizer_Multi(queues, [fm1, fm2], n_row=1, n_col=2)
        # vis = Visualizer_3D(queues, [fm0, fm2])
        # detector = Detector_Human()
        # vis = Visualizer_Cam_Data(
        #     queues, [fm0], detector=Detector_Human(min_prob=90), detector_start=0, save=True)
        # vis = Visualizer_Multi(queues, [fm0, fm1, fm2], n_row=1, n_col=3)
        # vis = Visualizer_Background(queues, [], save=True)
        vis = Visualizer_Single(queues, [])
        # vis = Visualizer_Base_2R(queues, [])
        vis.run(runflag)
    if num_radar == 2:
        fm01 = Frame_Manager_Base(max_length=6, ylim=[0, 3], zlim=[-1, 1])
        fm02 = Frame_Manager_Base(max_length=6, ylim=[0, 3], zlim=[-1, 1])

        vis = Visualizer_Base_2R(queues, [fm01], [fm02])
        vis.run(runflag)
 

def radar_thread(queue, runflag, radar):
    name, cfg_port, data_port, cfg_file = radar
    radar = Radar(name, cfg_port, data_port)
    radar.start(cfg_file, queue, runflag)
    

def main():
    radars = [radar_ports[i] for i in radar_to_use]
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
