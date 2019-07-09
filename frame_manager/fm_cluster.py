from .fm_base import Frame_Manager_Base
import numpy as np
from sklearn.cluster import DBSCAN

class Frame_Manager_Cluster(Frame_Manager_Base):
    def __init__(self, max_length=10, min_points=20):
        super().__init__(max_length)
        self.min_points = min_points

    def run(self, frame):
        if isinstance(frame, np.ndarray) and frame.shape[1] == 3:
            self.data.append(frame)

        out = np.concatenate(self.data)
        out = cluster_DBSCAN(out, self.min_points)
        if out is None:
            return np.asarray([[0, 0, 0]])
        return out



def cluster_DBSCAN(data, min_points):
    clusters = np.ndarray((0, 3))
    if not data.any() or data.shape[0] < 10:
        return None
    model = DBSCAN(eps=0.15)
    model.fit((data[:, :2]))
    labels = model.labels_
    n_cluster = np.unique(labels).shape[0]

    for _, class_idx in enumerate(np.unique(labels)):
        if class_idx != -1:
            class_data = data[labels == class_idx]
            if class_data.shape[0] < min_points:
                continue
            clusters = np.concatenate((clusters, class_data))

    return clusters
