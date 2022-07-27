from .fm_base import Frame_Manager_Base
import numpy as np
from sklearn.cluster import DBSCAN


class Frame_Manager_Motor(Frame_Manager_Base):
    def __init__(self, motorpos, max_length=1):
        super().__init__(max_length)
        self.motorpos = motorpos

    def run(self, frame):
        if isinstance(frame, np.ndarray) and frame.shape[1] >= 3:
            self.data.append(frame)
        off = self.motorpos.value/1000
        print(f'[motor_pos] {off:.4f}')
        out = np.concatenate(self.data)
        out[:,1] += off
        return out
