"""Application port first"""
RADAR_CFG = [
    ('1443A', 'COM4', 'COM3', './iwr1443/cfg/new.cfg'),       # 0
    ('1443B', 'COM10', 'COM9', './iwr1443/cfg/new.cfg'),     # 1
    ('1642', 'COM12', 'COM11', './iwr1642/cfg/heart.cfg'),    # 2
    ('6843A', 'COM10', 'COM9', './iwr6843/cfg/heart-cli.cfg'),  # 3
    ('6843B', 'COM17', 'COM16', './iwr6843/cfg/profileB.cfg'),   # 4
    ('1843', 'COM14', 'COM13', './iwr1843/cfg/heart-cli.cfg')   # 5
]

POLAR = {
    'H10': "c1:02:a3:7c:57:42",
    'PVS': "a0:9e:1a:ad:ee:48"
}

# MOTOR_PORT = 'COM11'

d_hor = 1.3
d_ver = 1.4

dp_hor = 0
dp_ver = 0
# dp_hor = 0.32
# dp_ver = 0.75

height_1 = 1.3
height_2 = 0.7
