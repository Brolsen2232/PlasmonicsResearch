import h5py
import numpy as np

def save_data(filename, type, data):
    with h5py.File(filename, 'w') as f:
        if type == 'fields':
            f.create_dataset('ez.r', data=np.real(data))  
            f.create_dataset('ez.i', data=np.imag(data))  
        elif type == 'geometries':
            for i, geom in enumerate(data):  
                group_name = f'geometry_{i}'
                group = f.create_group(group_name)

                if isinstance(geom, dict):  
                    for key, value in geom.items():
                        if isinstance(value, np.ndarray): 
                            group.create_dataset(key, data=value) 
                        else:
                            group.attrs[key] = str(value) 
                else:
                    raise ValueError("Expected geometry data to be a dictionary") 

def load_data(data_dir="data/"):
    geometries = []
    fields = None

    geometries_filename = data_dir + "geometries.h5"
    try:
        with h5py.File(geometries_filename, 'r') as f:
            for group_name in f:
                geometry_params = {}
                group = f[group_name]
                for key in group.attrs.keys():
                    geometry_params[key] = group.attrs[key]
                for dataset_name in group:
                    geometry_params[dataset_name] = group[dataset_name][()] 
                geometries.append(geometry_params)
    except FileNotFoundError:
        print("Geometries file not found.")

    fields_filename = data_dir + "fields.h5"
    try:
        with h5py.File(fields_filename, 'r') as f:
            ez_r = np.array(f['ez.r'])
            ez_i = np.array(f['ez.i'])
            fields = ez_r + 1j * ez_i
    except FileNotFoundError:
        print("Fields file not found.")

    return geometries, fields