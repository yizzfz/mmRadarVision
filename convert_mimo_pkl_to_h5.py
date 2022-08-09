"""Convert pkl data from demo_dca1000.py (with MIMO) to h5 data"""
import pickle
import numpy as np
import h5py

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

    # discard data at begining and end
    start = np.argmax(gts != None) + 5
    samples = raw.shape[3]
    n_chirp = raw.shape[2]
    n_rx = raw.shape[1]
    n_frame = raw.shape[0]
    frame_per_data = 1

    ind = np.arange(start+frame_per_data, n_frame - 5, frame_per_data, dtype=int)

    # create h5 data container
    h5fd = h5py.File(f'raw/{filename}.h5', 'w')
    h5fd.create_dataset('raw', shape=(len(ind), n_rx, frame_per_data *
                        n_chirp, samples), chunks=(1, n_rx, frame_per_data *
                        n_chirp, samples), dtype=complex)
    h5fd.create_dataset('gt', shape=(len(ind), frame_per_data), dtype=float)
    h5_raw = h5fd['raw']
    h5_gt = h5fd['gt']

    # start converting
    print(f'converting {len(ind)} data')
    for cnt, i in enumerate(ind):
        rawdata = raw[i-frame_per_data:i]
        rawdata = np.transpose(rawdata, (1, 0, 2, 3))
        rawdata = rawdata.reshape(n_rx, -1, samples)
        gt = gts[i-frame_per_data:i]
        h5_raw[cnt] = rawdata
        h5_gt[cnt] = np.asarray(gt, dtype=float)
        print(f'{cnt}/{len(ind)}', end='\r')
    print()
    h5fd.close()

if __name__ == "__main__":
    datasets = ['0727-1201-tmp']    # change this to your file name
    for f in datasets:
        convert_raw_to_raw(f)
