import numpy as np
from collections import deque

class Frame_Manager_Base():
    def __init__(self, max_length=10, xlim=None, ylim=None, zlim=None):
        self.data = deque([], max_length)
        self.xlim = xlim
        self.ylim = ylim
        self.zlim = zlim

    def run(self, frame):
        if isinstance(frame, np.ndarray) and frame.shape[1] == 3:
            self.data.append(frame)

        out = np.concatenate(self.data)
        out = self.filter(out)
        return out

    def filter(self, frame):
        if self.xlim:
            frame = frame[(frame[:, 0] >= self.xlim[0]) & (frame[:, 0] <= self.xlim[1])]
        if self.ylim:
            frame = frame[(frame[:, 1] >= self.ylim[0]) & (frame[:, 1] <= self.ylim[1])]
        if self.zlim:
            frame = frame[(frame[:, 2] >= self.zlim[0]) & (frame[:, 2] <= self.zlim[1])]
        return frame

