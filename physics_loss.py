import torch
from material import make_gold_material

def calculate_maxwell_residual(self, output, geometry_input, field_dimension, target_pattern=None):
    field_dim = field_dimension
    predicted_field = output
    lambd = 0.1
    gold = make_gold_material()
    espilon = gold.epsilon

    Ex_dy = torch.autograd.grad(predicted_field[:, 2], geometry_input, create_graph=True, allow_unused=True)[0][:, 1]  # dEz/dy
    Ex_dz = -torch.autograd.grad(predicted_field[:, 1], geometry_input, create_graph=True, allow_unused=True)[0][:, 2]  # -dEy/dz
    Ey_dz = torch.autograd.grad(predicted_field[:, 0], geometry_input, create_graph=True, allow_unused=True)[0][:, 2]  # dEx/dz
    Ey_dx = -torch.autograd.grad(predicted_field[:, 2], geometry_input, create_graph=True, allow_unused=True)[0][:, 0]  # -dEz/dx
    Ez_dx = torch.autograd.grad(predicted_field[:, 1], geometry_input, create_graph=True, allow_unused=True)[0][:, 0]  # dEy/dx
    Ez_dy = -torch.autograd.grad(predicted_field[:, 0], geometry_input, create_graph=True, allow_unused=True)[0][:, 1]  # -dEx/dy

    Hx_dt = torch.autograd.grad(predicted_field[:, 3], geometry_input, create_graph=True, allow_unused=True)[0][:, 2]  # dHz/dt
    Hy_dt = torch.autograd.grad(predicted_field[:, 4], geometry_input, create_graph=True, allow_unused=True)[0][:, 2]  # dHz/dt
    Hz_dt = torch.autograd.grad(predicted_field[:, 5], geometry_input, create_graph=True, allow_unused=True)[0][:, 2]  # dHz/dt

    residual_1 = Ey_dz - Ez_dy - epsilon * Ex_dt  # Faraday 1
    residual_2 = Ez_dx - Ex_dz - epsilon * Ey_dt  # Faraday 2
    residual_3 = Ex_dy - Ey_dx - epsilon * Ez_dt  # Faraday 3
    residual_4 = Hz_dy - Hy_dz  + epsilon * Hx_dt  # Ampere-Maxwell 1
    residual_5 = Hx_dz - Hz_dx  + epsilon * Hy_dt  # Ampere-Maxwell 2
    residual_6 = Hy_dx - Hx_dy  + epsilon * Hz_dt  # Ampere-Maxwell 3
    residual = torch.stack([residual_1, residual_2, residual_3, residual_4, residual_5, residual_6])
    return residual
   
def calculate_physics_loss(predicted_field, meep_field, cell_size, target_point, material):
    difference_real = meep_field.real - predicted_field.real 
    difference_imag = meep_field.imag - predicted_field.imag

    squared_magnitude_real = difference_real ** 2
    squared_magnitude_imag = difference_imag ** 2

    physics_loss_real = torch.mean(squared_magnitude_real)
    physics_loss_imag = torch.mean(squared_magnitude_imag)
    target_index = tuple(int(p * s) for p, s in zip(target_point, cell_size))
    field_enhancement_loss = -predicted_field[target_index]  

    return physics_loss_real + physics_loss_imag 