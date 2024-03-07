import torch 
import torch.nn as nn
from torch.optim import Adam

from geometry import generate_random_geometries
from field_prediction import FieldPredictionNetwork
from physics_loss import calculate_physics_loss
from simulation import run_simulation
from utils import save_data, load_data 

def train_model(model, geometries, fields, optimizer, num_epochs, target_point):
    for epoch in range(num_epochs):
        for (meep_geom, geom_params), field in zip(geometries, fields):
            optimizer.zero_grad()

            encoding = geom_params.get_encoding()

            input_tensor = torch.tensor(encoding, dtype=torch.complex64) 

            predicted_field = model(input_tensor)  
            loss = calculate_physics_loss(predicted_field, input_tensor, cell_size, target_point, material)    
            loss.backward() 
            optimizer.step() 

if __name__ == "__main__":
    cell_size = (1, 1, 1) 
    target_point = (0.5, 0.5, 0.5)  

    geometries = generate_random_geometries(cell_size, num_geometries=6)
    fields = [run_simulation(meep_geom, cell_size) for (meep_geom, geom_params) in geometries]    
    #save_data("data/geometries.h5", 'geometries', geometries)
    #save_data("data/fields.h5", 'fields', fields)

    geometry_dim = 6   
    field_dim = cell_size[0] * cell_size[1] * cell_size[2]  
    hidden_layers = [geometry_dim, geometry_dim]  

    material = 'gold'  # Or another dielectric/metal 

    model = FieldPredictionNetwork(geometry_dim, field_dim, hidden_layers)
    optimizer = Adam(model.parameters(), lr=1e-3)

    train_model(model, geometries, fields, optimizer, num_epochs=100, target_point=target_point) 
