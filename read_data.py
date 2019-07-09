import pickle
import pdb
import cv2

with open('./data/07081414.pkl', 'rb') as f:
    cam_data, radar_data = pickle.load(f) 

assert (len(cam_data) == len(radar_data))
cv2.namedWindow('cam')
for i, (cam, radar) in enumerate(zip(cam_data, radar_data)):
    cv2.imshow('cam', cam)
    cv2.waitKey()
