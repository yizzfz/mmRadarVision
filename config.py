"""
Radar name, port (application/user and data), and config file path, and rotation and translation vector
"""
RADAR_CFG = [
    ('1843', 'COM5', 'COM4', './radar_cfgs/1843.cfg', (0, 0, 0), (0, 0, 0)),
    # just some examples
    # ('1843', 'COM14', 'COM13', './radar_cfgs/1843.cfg', (0, 0, 135), (-1.2, 0.6, 1.5)),
    # ('1843', 'COM16', 'COM17', './radar_cfgs/1843.cfg', (0, 0, -140), (1.65, 3.45, 1.2)),
]

"""
If using Polar device
"""
# POLAR = {
#     'H10': "c1:02:a3:7c:57:42",
#     'PVS': "a0:9e:1a:ad:ee:48"
# }