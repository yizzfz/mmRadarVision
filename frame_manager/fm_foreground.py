from .fm_base import Frame_Manager_Base
import numpy as np

class Frame_Manager_Foreground(Frame_Manager_Base):
    def __init__(self, max_length=10, train_frame=200):
        super().__init__(max_length)
        self.step = 0
        self.noise_db = np.ndarray((0, 3))
        self.train_frame = train_frame

    def run(self, frame):
        print(self.step)
        if isinstance(frame, np.ndarray) and frame.shape[1] == 3:
            self.data.append(frame)
            if self.step < self.train_frame:
                self.noise_db = np.unique(np.concatenate(
                    (self.noise_db, np.round(frame, 1))), axis=0)
            self.step += 1
               
        if self.step < self.train_frame:
            return np.concatenate(self.data)

        out = np.concatenate(self.data)
        out_round = np.round(out, 1)
        idx = [i for i in range(out_round.shape[0]) if not (
            out_round[i] == self.noise_db).all(axis=1).any()]
        out_trimmed = out[idx]
        return out_trimmed
        
