"""
Radar name, port (application/user and data), and config file path
"""
RADAR_CFG = [
    ('1843', 'COM10', 'COM9', './radar_cfgs/1843.cfg', (0, 0, 45), (-1.2, -0.6, 1.5)),
    ('1843', 'COM14', 'COM13', './radar_cfgs/1843.cfg', (0, 0, 135), (-1.2, 0.6, 1.5)),
    # ('1843', 'COM10', 'COM9', './radar_cfgs/1843_raw.cfg'),
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
# from scipy.spatial.transform import Rotation as R
# radar_height1 = 1.5     # height of first radar
# radar_height2 = 1.5     # height of second radar
# d_hor = 1.3             # vertical distance between two radars
# d_ver = 1.4             # horizontal distance between two radars
# T1 = [0, d_ver, radar_height1]      # translation vector of the first radar
# T2 = [d_hor, 0, radar_height2]      # translation vector of the second radar
# # R1 = R.from_euler('z', 180, degrees=True).as_matrix()   # rotation matrix of the first radar
# # R2 = R.from_euler('z', -90, degrees=True).as_matrix()   # rotation matrix of the second radar
# R1 = [0, 0, 180]
# R2 = [0, 0, -90]