import h5py
import numpy as np

def load_data(data_dir="data/"):
    filenames = [data_dir + f for f in os.listdir(data_dir) if f.endswith(".h5")]
    geometries = [] 
    fields = []

    for filename in filenames:
        with h5py.File(filename, 'r') as f:
            fields.append(f['ez'][...])

    return geometries, np.array(fields)
def save_data(filename, fields):
    with h5py.File(filename, 'w') as f:
        f.create_dataset('ez', data=fields)  # Adjust the dataset name 'ez' if needed 