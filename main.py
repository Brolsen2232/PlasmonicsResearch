from geometry import ParameterizedGeometry, generate_random_geometry_params
from simulation import init_simulation, save_fields, calculate_field_enhancement, eV_to_meep_frequency
from neural_operator import FieldEnhancementOperator
from utils import load_data, save_data 
import torch 
import numpy as np


training_mode = "exploratory"  # Or "pattern_focused"


def generate_data(num_samples=50):
    data_dir = "data/"  
    geometries = []
    fields = []

    for i in range(num_samples):
       params = generate_random_geometry_params()
       para_geom = ParameterizedGeometry(params)
       geometry_params = para_geom.get_params()


       simulation = init_simulation(geometry_params, source_frequency)
       simulation.run(mp.at_end(save_fields), until=200) 
       field_enhancement = calculate_field_enhancement(simulation)
       print(f"Sample {i+1}: Field Enhancement = {field_enhancement:.3f}")

       fields.append(simulation.get_array(component=mp.Ez))
       geometries.append(para_geom.get_meep_geometry()) 

    save_data("data/geometries.h5", geometries)
    save_data("data/fields.h5", fields)

def train_model():
    geometries, fields = load_data("data/geometries.h5", "data/fields.h5") 

    model = FieldEnhancementOperator(geometry_dim=5, field_dim=(0.5, 0.5, 0.5, 3)) 
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(100):
         for geom, field in zip(geometries, fields):
             output = model(torch.tensor(geom).float())
             loss = model.physics_loss(output, torch.tensor(geom).float(), source_frequency) 

             optimizer.zero_grad()
             loss.backward()
             optimizer.step()

if __name__ == "__main__":
    source_frequency = eV_to_meep_frequency(2.34)  
    generate_data() 
    train_model()
