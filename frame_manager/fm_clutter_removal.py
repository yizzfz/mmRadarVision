from .fm_base import Frame_Manager_Base
import numpy as np
import warnings
warnings.simplefilter('once', RuntimeWarning)

class Frame_Manager_Clutter_Removal(Frame_Manager_Base):
    """Remove zero velocity points."""
    def __init__(self, *args, maxspeed=None, **kwargs):
        self.enable = True
        self.maxspeed = maxspeed
        super().__init__(*args, **kwargs)

    def run(self, frame):
        if isinstance(frame, np.ndarray) and frame.shape[1] >= 3:
            self.data.append(frame)

        out = np.concatenate(self.data)
        if self.enable:
            out = self.filter(out)
        return out

    def filter(self, frame):
        if frame.shape[1] < 4:
            warnings.warn('Invoked clutter removal module without collecting velocity info', RuntimeWarning)
            self.enable = False
            return frame
        frame = frame[frame[:, 3]!=0]
        if self.maxspeed is not None:
            frame = frame[frame[:, 3]<=self.maxspeed]
        return frame


def cluster_DBSCAN(data, min_points, axes):
    """DBSCAN algorithm.
    Parameters:
        min_points: minimal number of points in each cluster.
        axes: apply clustering along which dimensions. 
    """
    clusters = np.ndarray((0, 3))
    if not data.any() or data.shape[0] < 10:
        return np.empty((0, 3))
    model = DBSCAN(eps=0.15)
    model.fit((data[:, axes]))
    labels = model.labels_
    n_cluster = np.unique(labels).shape[0]

    for _, class_idx in enumerate(np.unique(labels)):
        if class_idx != -1:
            class_data = data[labels == class_idx]
            if class_data.shape[0] < min_points:
                continue
            clusters = np.concatenate((clusters, class_data))
    return clusters
