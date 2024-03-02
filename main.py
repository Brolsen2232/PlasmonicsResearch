from geometry import ParameterizedGeometry, generate_random_geometry_params
from simulation import init_simulation, calculate_field_enhancement, eV_to_meep_frequency
from neural_operator import FieldEnhancementOperator
from utils import load_data, save_data 
import torch 
import numpy as np
import meep as mp
import re

training_mode = "exploratory"  # Or "pattern_focused"


def generate_data(num_samples=5):
    data_dir = "data/"  
    geometries = []
    fields = []

    for i in range(num_samples):
       params = generate_random_geometry_params()
       para_geom = ParameterizedGeometry(params)
       geometry = para_geom.get_meep_geometry()
       


       simulation = init_simulation(geometry, source_frequency)
       simulation.run(mp.at_end(mp.output_efield_z), until=200) 
       field_enhancement = calculate_field_enhancement(simulation)
       print(f"Sample {i+1}: Field Enhancement = {field_enhancement:.3f}")

       fields.append(simulation.get_array(component=mp.Ez))
       geometries.append(params) 

    save_data("data/geometries.h5", 'geometries', geometries)  
    save_data("data/fields.h5", 'fields', fields)

def preprocess_geometry(geometry):
    center_numbers = re.findall(r"[-+]?\d*\.\d+|\d+", geometry['center'])
    geometry['center'] = np.array([float(num) for num in center_numbers])
    
    if '[' in geometry['size']:
        size_numbers = re.findall(r"[-+]?\d*\.\d+|\d+", geometry['size'])
        geometry['size'] = np.array([float(num) for num in size_numbers])
    else:
        geometry['size'] = np.array([float(geometry['size'])] * 3)  
    
    geometry['height'] = float(geometry['height'])
    geometry['radius'] = float(geometry['radius'])
    geometry['shape_type'] = float(geometry['shape_type'])

    return geometry

def train_model():
    geometries, fields = load_data("data/") 
    geometry_dim = 6
    out_features = 0.5 * 0.5 * 0.5 * 3  
    model = FieldEnhancementOperator(geometry_dim=5, field_dim=5) 
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(100):
         for unp_geom, field in zip(geometries, fields):
            input_tensor = torch.zeros(1, geometry_dim)
            geometry = preprocess_geometry(unp_geom)

            input_tensor[0, 0] = torch.tensor(geometry['shape_type'])  
            input_tensor[0, 1:4] = torch.tensor(geometry['size']) 
            input_tensor[0, 4:7] = torch.tensor(geometry['center']) 
            input_tensor[0, 7] = torch.tensor(geometry['radius']) 
            output = model(input_tensor.float())
            loss = model.physics_loss(output, torch.tensor(geometry).float(), source_frequency) 

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

if __name__ == "__main__":
    source_frequency = eV_to_meep_frequency(2.34)  
    generate_data() 
    train_model()
