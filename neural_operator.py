import torch  
import torch.nn as nn

class FieldEnhancementOperator(nn.Module):
    def __init__(self, geometry_dim, field_dim):
        super().__init__()

        self.linear1 = nn.Linear(geometry_dim, 64)
        self.fourier_modes = nn.Linear(64, 32)  # Parameterize Fourier modes
        self.linear2 = nn.Linear(32, field_dim)
        self.shape_embedding = nn.Embedding(3, 8)  # For embedding shape type
        self.linear0 = nn.Linear(geometry_dim + 8, 64) 

    def forward(self, geometry_input):
        shape_index = geometry_input[:, 0].long()  #
        shape_emb = self.shape_embedding(shape_index)
        x = torch.cat([geometry_input[:, 1:], shape_emb], dim=1)  #
        x = self.linear0(x)
        x = self.linear1(geometry_input)
        x = torch.sin(self.fourier_modes(x))  
        field_prediction = self.linear2(x)
        return field_prediction.reshape(-1, field_dim)  
    
    def get_relative_permittivity(self, geometry_input):
        grid_shape = self.field_dim.shape 
        epsilon = torch.ones(grid_shape)  # Default ε = 1

        center = geometry_input[0:3]
        size = geometry_input[3:6] 
        material_type = geometry_input[6] # Assuming some material encoding

        material_region = calculate_material_region(center, size) 

        if material_type == MaterialType.GOLD:
            epsilon[material_region] = self.gold_epsilon(source_frequency) 

        return epsilon

    def calculate_pattern_deviation(predicted_field, num_fourier_terms=3): 

        field_fft_x = torch.fft.fftn(predicted_field[:, 0, :]) # Example for Ex 
        field_fft_y =  torch.fft.fftn(predicted_field[0, :, :]) 
        field_fft_y =  torch.fft.fftn(predicted_field[::,:, 0]) 


        pattern_coefficients = extract_coefficients(field_fft_x, field_fft_y, field_fft_z,  num_fourier_terms)
        ideal_pattern = construct_ideal_pattern(pattern_coefficients)

        deviation = torch.mean((predicted_field - ideal_pattern)**2)
        return deviation 

    def physics_loss(self, output, geometry_input):
        predicted_field = output.reshape(-1, self.field_dim) 
        lambd = 0.1
        epsilon = self.get_relative_permittivity(geometry_input) 

        # 3D Time-Domain Maxwell's Equations 
        Ex_dy = anp.gradient(predicted_field[:, 2], 1) # dEz/dy
        Ex_dz = -anp.gradient(predicted_field[:, 1], 2) # -dEy/dz
        Ey_dz = anp.gradient(predicted_field[:, 0], 2) # dEx/dz
        Ey_dx = -anp.gradient(predicted_field[:, 2], 0) # -dEz/dx
        Ez_dx = anp.gradient(predicted_field[:, 1], 0) # dEy/dx
        Ez_dy = -anp.gradient(predicted_field[:, 0], 1) # -dEx/dy

        Hx_dt = anp.gradient(predicted_field[:, 3], 2) # dHz/dt
        Hy_dt = anp.gradient(predicted_field[:, 4], 2) # dHz/dt
        Hz_dt = anp.gradient(predicted_field[:, 5], 2) # dHz/dt


        residual_1 = Ey_dz - Ez_dy - epsilon * Ex_dt  # Faraday 1
        residual_2 = Ez_dx - Ex_dz - epsilon * Ey_dt  # Faraday 2
        residual_3 = Ex_dy - Ey_dx - epsilon * Ez_dt  # Faraday 3
        residual_4 = Hz_dy - Hy_dz  + epsilon * Hx_dt  # Ampere-Maxwell 1
        residual_5 = Hx_dz - Hz_dx  + epsilon * Hy_dt  # Ampere-Maxwell 2
        residual_6 = Hy_dx - Hx_dy  + epsilon * Hz_dt  # Ampere-Maxwell 3

                
        if training_mode == "exploratory":
            loss = torch.mean(residual**2)  # Focus solely on physics

        elif training_mode == "pattern_focused":
            pattern_deviation = calculate_pattern_deviation(predicted_field)
            loss = torch.mean(residual**2) + lambd * pattern_deviation

        else:
            raise ValueError("Invalid training mode")

        return loss
