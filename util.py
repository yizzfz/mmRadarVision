from sklearn.cluster import DBSCAN
import numpy as np
from config import *

out_dim = [60, 30, 60]
step = 0.01

def parse_radarcfg(radarcfg):
    with open(radarcfg, 'r') as f:
        cfg = f.readlines()
    for line in cfg:
        if line.startswith('%'):
            continue
        if 'profileCfg' in line:
            strs = line.split(' ')
            slope = float(strs[8])*1e12
            samples_per_chirp = int(strs[10])
            ADC_rate = float(strs[11])*1e3
        if 'frameCfg' in line:
            strs = line.split(' ')
            chirps_per_frame = int(strs[3])
            frame_time = float(strs[5])/1e3
    config = {
        'samples_per_chirp': samples_per_chirp,
        'chirps_per_frame': chirps_per_frame,
        'slope': slope,
        'ADC_rate': ADC_rate,
        'frame_time': frame_time,
        'fps': chirps_per_frame / frame_time,
    }
    return config

# input (n, 3), output [(n,3), ...]
def cluster_DBSCAN(data, min_points=5, eps=0.1, ret_centroids=False):
    assert(len(data.shape)==2 and data.shape[1]==3)
    if not data.any() or data.shape[0] < 10:
        return None
    model = DBSCAN(eps=eps)
    model.fit((data[:, :2]))
    labels = model.labels_
    clusters = []
    centroids = []

    for _, class_idx in enumerate(np.unique(labels)):
        if class_idx != -1:
            class_data = data[labels == class_idx]
            if class_data.shape[0] < min_points:
                continue

            clusters.append((class_data))
            centroids.append(np.average(class_data, axis=0))
    if ret_centroids:
        return clusters, centroids
    return clusters

def frame_to_mat(frame, out_dim=out_dim, step=step, ret_centroids=False):
    res = []
    if ret_centroids:
        clusters, centroids = cluster_DBSCAN(frame, ret_centroids=ret_centroids)
    else:
        clusters = cluster_DBSCAN(frame)

    if len(clusters) == 0:
        return None
    for i, obj in enumerate(clusters):
        mat = obj_to_mat(obj, out_dim, step)
        res.append(mat)
    res = np.stack(res)
    if ret_centroids:
        return res, centroids
    return res


def obj_to_mat(obj, out_dim=out_dim, step=step):
    obj = np.round(obj, 2)
    obj = obj - np.min(obj, axis=0)
    obj = np.asarray(obj/step, dtype=np.uint8)
    mat = np.zeros(out_dim, dtype=np.uint8)
    for p in obj:
        try:
            mat[tuple(p)] = 1
        except IndexError:
            continue
    return mat