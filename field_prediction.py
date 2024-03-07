import torch.nn as nn
import torch
import torch.nn as nn
import torch

class FieldPredictionNetwork(nn.Module):
    def __init__(self, geometry_dim, field_dim, hidden_layers):
        super().__init__()
        self.output_linear = nn.Linear(geometry_dim, geometry_dim, dtype=torch.complex64) 
        self.fourier_blocks = nn.ModuleList()
        for _ in range(2):  
            self.fourier_blocks.append(FourierBlock())
        self.intermediate_layer = nn.Linear(geometry_dim, 32, dtype=torch.complex64) 
        self.output_layer = nn.Linear(hidden_layers[-1], field_dim, dtype=torch.complex64)  

    def forward(self, geometry_input):
        geometry_input = geometry_input.unsqueeze(0)  

        x = self.output_linear(geometry_input) 
        
        for block in self.fourier_blocks:
            x = block(x)
        x = self.intermediate_layer(x)
        field_prediction = self.output_layer(x)
        return field_prediction


class FourierBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(7,7)
        self.learned_kernel = nn.Parameter(torch.randn(3, 3, 3, 1))  
 

    def forward(self, x):
        x_ft = torch.fft.fftn(x)
        x_ft = x_ft * self.learned_kernel 
        x = torch.fft.ifftn(x_ft)
        return x 

