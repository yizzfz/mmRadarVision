import cv2
import numpy as np
import pickle
import pdb
import matplotlib.pyplot as plt
import random
import sys
import datetime
import os
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))
from detector import Detector_Human


data_file = '07091724.pkl'


def main():
    data = read_data(data_file)
    print(f'Found {len(data)} frames')
    # n = random.randint(20, len(data))
    # frame = data[n]

    detector = Detector_Human(min_prob=60)
    processed_data = process_frame(data, detector)
    save_data(processed_data)


def process_frame(data, detector):
    res = []
    for i in range(len(data)):
        frame = data[i]
        ret, boxes = detector.process(np.stack([frame[0]]*3, axis=2))
        res.append((boxes, ret, frame[1:]))
        print(f'{i}/{len(data)}')
    return res


def read_data(data_file):
    with open(data_file, 'rb') as f:
        data = pickle.load(f)
    return data


def save_data(data):
    # timestamp = datetime.datetime.now().strftime('%m%d%H%M')
    with open(f'pp-{data_file}', 'wb') as f:
        pickle.dump(data, f)


if __name__ == '__main__':
    main()
