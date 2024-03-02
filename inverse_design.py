from neural_operator import FieldEnhancementOperator
from geometry import ParameterizedGeometry
import torch 

def inverse_design(model, target_enhancement, init_geometry):
    geometry_params = init_geometry
    optimizer = torch.optim.Adam(geometry_params.parameters(), lr=...) 

    for iter in range(max_iterations):
        output = model(torch.tensor(geometry_params).float())
        field_enhancement = calculate_field_enhancement_from_prediction(output)

        # Calculate gradient of field enhancement w.r.t. geometry_params

        optimizer.zero_grad()
        loss = (field_enhancement - target_enhancement)**2 
        loss.backward()
        optimizer.step()


    return geometry_params 
