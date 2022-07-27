import pickle
import numpy as np
import h5py
fft_freq = np.fft.fftfreq(6000, d=1.0/9e6)
fft_freq_d = fft_freq*3e8/2/21e12
max_freq_i = np.argmax(fft_freq_d > 3)

def convert_raw_to_raw(filename):
    print('loading', filename)
    with open(f'{filename}.pkl', 'rb') as f:
        header = pickle.load(f)
        print(header)
        # assuming there are two types of data
        raw = pickle.load(f)
        gts = pickle.load(f)

    # convert to numpy array
    raw = np.asarray(raw)
    gts = np.asarray(gts)

    if len(raw.shape) == 4:
        raw = raw[:, 0, :, :]

    # discard data at begining
    start = np.argmax(gts != None) + 5
    samples = raw.shape[2]
    n_chirp = raw.shape[1]
    n_frame = raw.shape[0]
    frame_per_data = 1

    ind = np.arange(start+frame_per_data, n_frame - 5, frame_per_data, dtype=int)

    # create h5 data container
    h5fd = h5py.File(f'raw/{filename}.h5', 'w')
    h5fd.create_dataset('raw', shape=(len(ind), frame_per_data *
                        n_chirp, samples), chunks=(1, frame_per_data *
                        n_chirp, samples), dtype=complex)
    h5fd.create_dataset('gt', shape=(len(ind), frame_per_data), dtype=float)
    h5_raw = h5fd['raw']
    h5_gt = h5fd['gt']

    # start converting
    print(f'converting {len(ind)} data')
    for cnt, i in enumerate(ind):
        fft = raw[i-frame_per_data:i].reshape((-1, samples))
        gt = gts[i-frame_per_data:i]
        h5_raw[cnt] = fft
        h5_gt[cnt] = np.asarray(gt, dtype=float)
        print(f'{cnt}/{len(ind)}', end='\r')
    print()
    h5fd.close()

if __name__ == "__main__":
    datasets = ['0000-0000']    # change this to your file name
    for f in datasets:
        convert_raw_to_raw(f)
