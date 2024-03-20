import torch.nn as nn
import torch



class FieldPredictionNetwork(nn.Module):
    def __init__(self, geometry_dim, field_dim, hidden_layers):
        super().__init__()

        self.input_layer = nn.Linear(geometry_dim, hidden_layers[0])  
        self.interm_layers = nn.ModuleList([
            nn.Linear(hidden_layers[i], hidden_layers[i+1]) 
            for i in range(len(hidden_layers) - 1)
        ])
        self.fno_block = FNOBlock(modes=32, width=64) 
        self.output_layer = nn.Linear(hidden_layers[-1], field_dim) 

    def forward(self, geometry_input):
        x = self.input_layer(geometry_input) 
        x = torch.relu(x)  

        for layer in self.interm_layers:
            x = torch.relu(layer(x))

        x = self.fno_block(x)

        field_prediction = self.output_layer(x)
        return field_prediction


class FNOBlock(nn.Module):
    def __init__(self, modes, width):
        super().__init__()
        self.modes = modes
        self.width = width
        self.weights = nn.ParameterList([nn.Parameter(torch.randn(modes, width)) for _ in range(2)]) 
        self.lin1 = nn.Linear(width, width)
        self.lin2 = nn.Linear(width, width)


    def forward(self, x):
        
        x = torch.fft.fftn(x)
        L = torch.diag(torch.exp(1j * 2 * np.pi * torch.arange(self.modes) / self.modes)) 

        x = x.unsqueeze(-1) @ self.weights[0] 
        x = torch.relu(x)
        x = L @ x  
        x = x @ self.weights[1]
        x = self.lin1(x.squeeze(-1))
        x = torch.relu(x)
        x = self.lin2(x)
        x = torch.fft.ifftn(x)
        return x