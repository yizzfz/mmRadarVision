# mmRadarVision

This project uses mmWave radars for some computer vision tasks. It supports real-time data streaming, processing and visualization from a Texas Instruments (TI) mmWave radar board to a PC. 
It can be used for either point cloud data (with the TI out-of-box demo firmware) or raw data (with the TI DCA1000EVM board and an appropriate firmware).

## Prerequisite 

Install Python (tested with 3.9) and install dependencies with ```pip install -r requirement.txt```.

## To collect point cloud data

This project allows point cloud capturing from the radar.
Before running this project, it is strongly recommended to test the hardware setup using the TI mmWave Demo Visualizer ([Instruction](https://www.ti.com/lit/ug/swru587/swru587.pdf)).
This ensures that the radar board is working properly with the correct firmware, and the drivers are installed. 
Or install the [standalone xds110 driver](https://software-dl.ti.com/ccs/esd/documents/xdsdebugprobes/emu_xds_software_package_download.html).
1. Check device manager to find the COM port (User & Data) of the radar. 
2. Modify config.py to specify the COM port and radar location. 
3. Modify demo.py to select radar (and order).
4. Run ```python demo.py```.

## To collect raw data with DCA1000

This project allows raw data capturing from the radar using the TI DCA1000EVM board, bypassing the mmWave studio software. 
The backend of this project uses a modified version of the TI DCA1000CLI tool.
Before running this project, it is strongly recommended to test the hardware setup using the TI mmWave studio ([Instruction](https://www.ti.com/lit/ml/spruik7/spruik7.pdf)).
This ensures that the radar board and the DCA1000EVM are working properly with the correct firmware, and the drivers are installed. 
1. Check device manager to find the COM port (User & Data) of the radar. 
2. Modify config.py to specify the COM port. 
3. Configure the IP of Ethernet adaptor to static: 192.168.33.30.
4. Modify ```fileBasePath``` in dca1000/dca1000.json,
5. Modify demo_dca1000.py to select radar.
6. Run ```python demo_dca1000.py```.

## Citation

If you find this work useful, please consider cite:
```
@article{Cui21,
  author={Cui, Han and Dahnoun, Naim},
  journal={IEEE Aerospace and Electronic Systems Magazine}, 
  title={High Precision Human Detection and Tracking Using Millimeter-Wave Radars}, 
  year={2021},
  volume={36},
  number={1},
  pages={22-32},
  doi={10.1109/MAES.2020.3021322}}
```
