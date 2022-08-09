from .fm_base import Frame_Manager_Base
import numpy as np

class Frame_Manager_Foreground(Frame_Manager_Base):
    """Perform foreground extraction. 
    Remembers the location of the clutters in the train_frames and substracts them from incoming frames."""
    def __init__(self, max_length=10, train_frame=200, axes=(0, 1, 2)):
        """
        Parameters:
            max_length: number of frames to stack.
            train_frame: use the first `train_frame` frames to remember clutters.
            axes: dimensions to use, default x-y-z.
        """
        super().__init__(max_length)
        self.step = 0
        self.noise_db = np.ndarray((0, 3))
        self.train_frame = train_frame
        self.axes = axes

    def run(self, frame):
        print(self.step)
        if isinstance(frame, np.ndarray) and frame.shape[1] >= 3:
            self.data.append(frame)
            if self.step < self.train_frame:    # remember cluters
                self.noise_db = np.unique(np.concatenate(
                    (self.noise_db, np.round(frame[:, self.axes], 1))), axis=0)
            self.step += 1
               
        if self.step < self.train_frame:
            return np.concatenate(self.data)

        # after training, substract clutters from incoming frames
        out = np.concatenate(self.data)
        out_round = np.round(out, 1)[:, self.axes]
        idx = [i for i in range(out_round.shape[0]) if not (
            out_round[i] == self.noise_db).all(axis=1).any()]
        out_trimmed = out[idx]
        return out_trimmed
        
