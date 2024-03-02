import meep as mp
import numpy as np
from material import make_gold_material  

class ParameterizedGeometry:
    def __init__(self, params):
         self.params = params

    def __getitem__(self, key):
        return self.params.get(key) 

    def get_meep_geometry(self):
        geometry = []

        if 'shape_type' in self.params:
            shape_type = self.params['shape_type']
            center = mp.Vector3(*self.params['center'])

            if shape_type == 0:  # Box
                size = mp.Vector3(*self.params['size'])
                geometry.append(mp.Block(size=size, center=center, material=make_gold_material()))
 
            elif shape_type == 1:  # Cylinder
                radius = self.params['radius']
                height = self.params['height']
                geometry.append(mp.Cylinder(radius=radius, height=height, center=center, axis=mp.Vector3(0, 0, 1), material=make_gold_material()))

            elif shape_type == 2:  # Sphere
                radius = self.params['radius']
                geometry.append(mp.Sphere(radius=radius, center=center, material=make_gold_material()))

            else:
                raise ValueError("Unsupported shape type")

def generate_box_params():
    params = {
            'shape_type': 0,  # Now with integer encoding
            'size': np.random.uniform(0.01, 0.2, size=3),
            'center': np.random.uniform(-0.2, 0.2, size=3),
            'radius': 0,
            'height': 0
        }
    return params

def generate_cylinder_params():
    params = {
            'shape_type': 1, 
            'size': 0,
            'height': np.random.uniform(0.05, 0.15),
            'radius': np.random.uniform(0.01, 0.1), 
            'center': np.random.uniform(-0.2, 0.2, size=3) 
        }
    return params

def generate_sphere_params():
    params = {
            'shape_type': 2, 
            'size': 0,
            'radius': np.random.uniform(0.05, 0.1), 
            'center': np.random.uniform(-0.2, 0.2, size=3), 
            'height': 0
        }
    return params

def generate_random_geometry_params():
    num_objects = np.random.randint(1, 20)
    geometries = {} 

    for i in range(num_objects):  
        shape_choice = np.random.choice(['box', 'cylinder', 'sphere'])

        if shape_choice == 'box':
            geometries[f'geometry_{i}'] = generate_box_params()  # Unique keys
        elif shape_choice == 'cylinder':
            geometries[f'geometry_{i}'] = generate_cylinder_params()
        else:
            geometries[f'geometry_{i}'] = generate_sphere_params()

    return geometries 