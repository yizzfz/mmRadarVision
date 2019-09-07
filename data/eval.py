'''
input - a list as [cam res, two radar res, one radar res] (output from visualizer)
'''


import numpy as np
import pickle
import sys
import matplotlib.pyplot as plt
from scipy.signal import medfilt
import os


data_folder = 'two-radar'


def main():
    pkl_list = get_pkl_list(data_folder)
    for pkl in pkl_list:
        data = read_data(pkl)
        data = np.array(data)

        # data[:, 0] = medfilt(data[:, 0], 5)

        # plt.plot(data[:, 0])
        # plt.plot(data[:, 1])
        # plt.plot(data[:, 2])
        # plt.show()

        tp1 = np.minimum(data[:, 0], data[:, 1])
        tp2 = np.minimum(data[:, 0], data[:, 2])

        # sensitivity / hit rate = true positive / real postive
        s1 = np.average(np.nan_to_num(tp1 / data[:, 0], nan=1))
        s2 = np.average(np.nan_to_num(tp2 / data[:, 0], nan=1))

        # precision = true postive / predicted postive
        p1 = np.average(np.nan_to_num(tp1 / data[:, 1], nan=1))
        p2 = np.average(np.nan_to_num(tp2 / data[:, 2], nan=1))
        
        print(f'one radar hit rate {s2:.4f}, precision {p2:.4f}, two radar hit rate {s1:.4f}, precision {p1:.4f}')
        import pdb; pdb.set_trace()


def get_pkl_list(folder):
    res = []
    for root, dirs, files in os.walk(folder, topdown=False):
        for name in files:
            if name.endswith('.pkl'):
                res.append(os.path.join(root, name))
            
    print(res)
    return res

def read_data(data_file):
    with open(data_file, 'rb') as f:
        data = pickle.load(f)
    return data
        
if __name__ == '__main__':
    main()
