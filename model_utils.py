import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

class FCNN(nn.Module):
    def __init__(self, input_dim: int, hidden_layers: List[int], output_dim: int, dropout_rate: float = 0.5):
        super(FCNN, self).__init__()
        layers = []
        current_dim = input_dim
        activations = [nn.ReLU(), nn.Tanh(), nn.LeakyReLU(0.1)]  # Different activation functions

        for i, hidden_dim in enumerate(hidden_layers):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(activations[i % len(activations)])  # Cycle through different activations
            layers.append(nn.Dropout(dropout_rate))
            current_dim = hidden_dim
        
        layers.append(nn.Linear(current_dim, output_dim))
        self.model = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        return self.model(x)








