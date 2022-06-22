"""
Radar name, port (application/user and data), and config file path
"""
RADAR_CFG = [
    ('1843', 'COM4', 'COM3', './radar_cfgs/iwr1843/cfg/profile.cfg'),
    ('1843', 'COM4', 'COM3', './radar_cfgs/iwr1843/cfg/profile_dca1000.cfg'),
]

"""
If using Polar device
"""
POLAR = {
    'H10': "c1:02:a3:7c:57:42",
    'PVS': "a0:9e:1a:ad:ee:48"
}

"""
If two radars are used, defined their location
"""
from scipy.spatial.transform import Rotation as R
radar_height1 = 1.5     # height of first radar
radar_height2 = 1.5     # height of second radar
d_hor = 1.3             # vertical distance between two radars
d_ver = 1.4             # horizontal distance between two radars
T1 = [0, d_ver, radar_height1]  
T2 = [d_hor, 0, radar_height2]
R1 = R.from_euler('z', 180, degrees=True).as_matrix()
R2 = R.from_euler('z', -90, degrees=True).as_matrix()
