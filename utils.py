import h5py
import meep as mp
import numpy as np
from geometry import ParameterizedGeometry

def save_data(filename, type, data):
    with h5py.File(filename, 'w') as f:
        if type == 'fields':
            f.create_dataset('ez.i', data=np.imag(data))
            f.create_dataset('ez.r', data=np.real(data))
        elif type == 'geometries':
            for i, geom in enumerate(data):
                group_name = f'geometry_{i}'
                group = f.create_group(group_name)
                for key, value in geom.items():
                    if isinstance(value, dict):
                        sub_group = group.create_group(key)
                        for inner_key, inner_value in value.items():
                            sub_group.attrs[inner_key] = str(inner_value)
                    else:
                        group.attrs[key] = str(value)

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
                for sub_group_name in group:
                    sub_group = group[sub_group_name]
                    for key in sub_group.attrs.keys():
                        geometry_params[key] = sub_group.attrs[key]
                geometries.append(geometry_params)
    except FileNotFoundError:
        print("Geometries file not found.")

    fields_filename = data_dir + "fields.h5"
    try:
        with h5py.File(fields_filename, 'r') as f:
            ez_i = np.array(f['ez.i'])
            ez_r = np.array(f['ez.r'])
            fields = ez_r + 1j * ez_i
    except FileNotFoundError:
        print("Fields file not found.")

    return geometries, fields
