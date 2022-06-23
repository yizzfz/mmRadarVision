# mmRadarVision
Using mmWave radars for some computer vision tasks. 

## Installation 
Install Python (tested with 3.9) and install dependencies with ```pip install -r requirement.txt```.

Before running the script, try to operate the radar with TI mmWave Demo Visualizer.

## To collect point cloud data
1. Check device manager to find the COM port (User & Data) of the radar. 
2. Modify config.py to specify the COM port and radar location. 
3. Modify demo.py to select radar (and order).
4. Run ```python demo.py```.

## To collect raw data with DCA1000
1. Check device manager to find the COM port (User & Data) of the radar. 
2. Modify config.py to specify the COM port. 
3. Configure the IP of Ethernet adaptor to static: 192.168.33.30.
4. Modify demo_dca1000.py to select radar.
4. Run ```python demo_dca1000.py```.
