import torch
import torch.nn as nn
class SGR(nn.Module):
    def __init__(self, input_dim):
        super(SGR, self).__init__()
        self.model = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, 64),  # Reduced from 128 to 64
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
