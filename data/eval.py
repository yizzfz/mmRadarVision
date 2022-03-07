'''
input - a list as [cam res, two radar res, one radar res] (output from visualizer)
'''


import numpy as np
import pickle
import sys
import matplotlib.pyplot as plt
from scipy.signal import medfilt
import os

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


data_folder = 'two-radar-tmp1'


def main():
    pkl_list = get_pkl_list(data_folder)
    data_all = [np.array(read_data(pkl))[:, 0:3] for pkl in pkl_list]
    for data in data_all:
    #     # data[:, 0] = medfilt(data[:, 0], 5)

    #     # plt.plot(data[:, 0])
    #     # plt.plot(data[:, 1])
    #     # plt.plot(data[:, 2])
    #     # plt.show()
        process_data(data)
        # cm(data)
    process_data(np.concatenate(data_all))
    cm(np.concatenate(data_all))

def cm(data):
    data[:, 0] = medfilt(data[:, 0], 3)
    data[:, 1] = medfilt(data[:, 1], 3)
    # data[data>3] = 3

    print(confusion_matrix(data[:, 0], data[:, 1]))
    print(accuracy_score(data[:, 0], data[:, 1]))
    print(classification_report(data[:, 0], data[:, 1]))

    for i in range(0, 5):
        idx = (data[:, 0] == i)
        prediction = data[idx, 1]
        # prediction = np.random.choice(prediction, 1000)
        for j in range(0, 5):
            print(np.sum(prediction==j), end=' ')
        print()

def process_data(data):
    data[:, 0] = medfilt(data[:, 0], 5)
    tp1 = np.minimum(data[:, 0], data[:, 1])
    tp2 = np.minimum(data[:, 0], data[:, 2])

    # sensitivity / hit rate = true positive / real postive
    s1 = np.average(np.nan_to_num(tp1 / data[:, 0], nan=1))
    s2 = np.average(np.nan_to_num(tp2 / data[:, 0], nan=1))

    # precision = true postive / predicted postive
    p1 = np.average(np.nan_to_num(tp1 / data[:, 1], nan=1))
    p2 = np.average(np.nan_to_num(tp2 / data[:, 2], nan=1))


    print(f'[{data.shape}] one radar hit rate {s2:.4f}, precision {p2:.4f}, two radar hit rate {s1:.4f}, precision {p1:.4f}')

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
