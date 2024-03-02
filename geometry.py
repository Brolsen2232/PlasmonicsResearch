import meep as mp
import numpy as np


class ParameterizedGeometry:
    def __init__(self, params):
        self.params = params

    def get_params(self):
        return self.params

    def get_meep_geometry(self):
        geometry = []
        if 'shape_type' in self.params:
            if self.params['shape_type'] == 'box':
                size = mp.Vector3(*self.params['size'])
                center = mp.Vector3(*self.params['center'])
                geometry.append(mp.Block(size=size, center=center))
 
            elif self.params['shape_type'] == 'cylinder':
                radius = self.params['radius']
                height = self.params['height']
                center = mp.Vector3(*self.params['center'])
                geometry.append(mp.Cylinder(radius=radius, height=height, center=center, axis=mp.Vector3(0, 0, 1)))
       
            elif self.params['shape_type'] == 'sphere':
                radius = self.params['radius']
                center = mp.Vector3(*self.params['center'])
                geometry.append(mp.Sphere(radius=radius, center=center))

            else:
                raise ValueError("Unsupported shape type")

        return geometry 

    #def get_material_regions(self):

def generate_box_params():
    params = {
            'shape_type': 'box',
            'size': np.random.uniform(0.05, 0.2, size=3),
            'center': np.random.uniform(-0.2, 0.2, size=3)
        }
    return params

def generate_cylinder_params():
    params = {
            'shape_type': 'cylinder',
            'height': np.random.uniform(0.05, 0.15),
            'radius': np.random.uniform(0.02, 0.05), 
            'center': np.random.uniform(-0.2, 0.2, size=3) 
        }
    return params

def generate_sphere_params():
    params = {
            'shape_type': 'sphere',
            'radius': np.random.uniform(0.05, 0.1), 
            'center': np.random.uniform(-0.2, 0.2, size=3)
        }
    return params

def generate_random_geometry_params():
    num_objects = np.random.randint(1, 5)
    params = [] 

    for _ in range(num_objects):
        shape_choice = np.random.choice(['box', 'cylinder', 'sphere'])

        if shape_choice == 'box':
            params.append(generate_box_params())
        elif shape_choice == 'cylinder':
            params.append(generate_cylinder_params())
        else:
            params.append(generate_sphere_params())

    return params