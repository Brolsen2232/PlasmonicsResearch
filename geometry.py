import numpy as np
import meep as mp 
from material import make_gold_material

class Geometry:
    SHAPE_TYPES = {'box': 0, 'cylinder': 1, 'sphere': 2}  
    def __init__(self, shape_type, center, material='gold', **kwargs):
        self.shape_type = self.SHAPE_TYPES[shape_type]
        self.center = center
        self.material = material
        self.size = kwargs.get('size')
        self.radius = kwargs.get('radius')
        self.height = kwargs.get('height')

    def get_encoding(self):  
        encoding_parts = [self.shape_type, *self.center]  

        if self.size is not None:
            encoding_parts.extend(self.size)
        if self.radius is not None:
            encoding_parts.append(self.radius)
        if self.height is not None:
            encoding_parts.append(self.height)

        return np.array(encoding_parts) 

    def get_meep_geometry(self, cell_size):
        meep_geometry = []
        if self.shape_type == 0:  # Box
            meep_geometry.append(mp.Block(size=mp.Vector3(*self.size), 
                                          center=mp.Vector3(*self.center), 
                                          material=make_gold_material()))
        elif self.shape_type == 1:  # Cylinder
            meep_geometry.append(mp.Cylinder(radius=self.radius, 
                                             height=self.height,
                                             center=mp.Vector3(*self.center), 
                                             axis=mp.Vector3(0, 0, 1), 
                                             material=make_gold_material()))
        elif self.shape_type == 2:  # Sphere
            meep_geometry.append(mp.Sphere(radius=self.radius,
                                           center=mp.Vector3(*self.center), 
                                           material=make_gold_material()))
        else:
            raise ValueError("Unsupported shape type")

        return meep_geometry

def generate_random_geometries(cell_size, num_geometries):
    geometries = []  # Store pairs of (meep_geometry, geometry_params)
    for _ in range(num_geometries):
        shape_choice = np.random.choice(['box', 'cylinder', 'sphere'])
        params = get_shape_parameters(shape_choice, cell_size)  
        geom_param = Geometry(shape_type=shape_choice, **params)

        meep_geometry = geom_param.get_meep_geometry(cell_size)
        geometries.append((meep_geometry, geom_param))

    return geometries 

def get_shape_parameters(shape_type, cell_size):
    params = {}
    if shape_type == 'box':
        params['size'] = np.random.uniform(0.05, 0.2, size=3)
        params['center'] = get_random_center(cell_size, params['size'])
    elif shape_type == 'cylinder':
        params['radius'] = np.random.uniform(0.02, 0.1)
        params['height'] = np.random.uniform(0.05, 0.2)
        params['center'] = get_random_center(cell_size, radius=params['radius'])  
    elif shape_type == 'sphere':
        params['radius'] = np.random.uniform(0.02, 0.1)
        params['center'] = get_random_center(cell_size, radius=params['radius'])  
    else:
        raise ValueError('Unsupported shape type')

    return params

def get_random_center(cell_size, size=None, radius=None):
    margin = 0.05 
    max_coords = np.array(cell_size) - margin 
    min_coords = np.array([margin, margin, margin]) 
    if size is not None:
        max_coords -= size / 2
        min_coords += size / 2
    elif radius is not None:
        max_coords -= radius 
        min_coords += radius 

    return np.random.uniform(min_coords, max_coords) 
