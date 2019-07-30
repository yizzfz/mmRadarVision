'''
input - a dict as {'True':[list of point cloud], ...} (output from visualizer)
output - a dict as {'Ture': ndarray(n, out_dim), ...}
'''


import numpy as np
import pickle
import sys
sys.path.append("..")
from util import frame_to_mat, obj_to_mat, out_dim, step

data_file = '07261815.pkl'


def main():
    data = read_data(data_file)
    res = dict()
    print('start')
    for key, objs in data.items():
        mats = process_data(objs)
        print(f'[{key}] done')
        res[key] = mats
    save_data(res)


def process_data(objs):
    res = []
    for i, obj in enumerate(objs):
        print(f'{i}/{len(objs)}', end='\r')
        mat = obj_to_mat(obj, out_dim, step)
        res.append(mat)
    res = np.stack(res)
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
