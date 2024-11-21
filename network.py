import torch 
import torch.nn as nn

class ReLUk(nn.Module):
    def __init__(self, k=1):
        super().__init__()
        self.k = k 
    
    def forward(self, x):
        return torch.maximum(torch.zeros_like(x), x) ** self.k
    
# A simple feedforward neural network
class DeepNN(torch.nn.Module):
    def __init__(self, layers, nonlinearity, aug=None):
        super(DeepNN, self).__init__()

        self.n_layers = len(layers) - 1
        self.aug = aug

        assert self.n_layers >= 1

        self.layers = nn.ModuleList()

        for j in range(self.n_layers):
            self.layers.append(nn.Linear(layers[j], layers[j+1]))
            if j != self.n_layers - 1:
                self.layers.append(nonlinearity)

    def forward(self, x):
        for _, l in enumerate(self.layers):
            x = l(x)

        return x