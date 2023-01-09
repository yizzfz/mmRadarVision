from sklearn.cluster import DBSCAN
import numpy as np
from config import *

out_dim = [60, 30, 60]
step = 0.01

def parse_radarcfg(radarcfg):
    """Parse the key parameters in the radar configuration file"""
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
            chirploops_per_frame = int(strs[3])
            chirps_per_loop = int(strs[2]) - int(strs[1]) + 1
            chirps_per_frame = chirploops_per_frame * chirps_per_loop
            frame_time = float(strs[5])/1e3
        if 'channelCfg' in line:
            strs = line.split(' ')
            n_rx = f'{int(strs[1]):04b}'.count('1')
            n_tx = f'{int(strs[2]):03b}'.count('1')
    config = {
        'samples_per_chirp': samples_per_chirp,
        'chirps_per_frame': chirps_per_frame,
        'slope': slope,
        'ADC_rate': ADC_rate,
        'frame_time': frame_time,
        'fps': chirps_per_frame / frame_time,
        'n_rx': n_rx,
        'n_tx': n_tx,
        'chirploops_per_frame': chirploops_per_frame,
        'chirps_per_loop': chirps_per_loop
    }
    return config

def cluster_DBSCAN(data, min_points=5, distance=0.1, ret_centroids=False):
    """DBSCAN algorithm, input (n, 3), output [(n,3), ...]

    Parameters:
        data: (n, 3) point cloud.
        min_points: the minimal number of points in a cluster.
        distance: the maximum distance between points in the same cluster.
        ret_centroids: return the centroids of the clsuters along with the clusters
    """
    assert(len(data.shape)==2 and data.shape[1]==3)
    if not data.any() or data.shape[0] < 10:
        return None
    model = DBSCAN(eps=distance)
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
    """Clsuter and voxelize a point cloud into a dense matrix"""
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
    """Voxelize a point cloud into a dense matrix"""
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

def nchannel_from_dataformat(outformat):
    """Mask output matrix based on configured data format"""
    m = []
    if 'p' in outformat:
        m += [0, 1, 2]
    if 'v' in outformat:
        m.append(3)
    if 's' in outformat:
        m.append(4)
    if 'n' in outformat:
        m.append(5)
    return len(m)