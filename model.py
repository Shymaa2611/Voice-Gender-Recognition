import torch
import torch.nn as nn

class SGR(nn.Module):
    def __init__(self, input_dim):
        super(SGR, self).__init__()
        self.model = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, 128),  # Increased from 100 to 128
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(128, 256),        # Increased from 200 to 256
            nn.ReLU(),
            nn.Dropout(0.5),
            
            nn.Linear(256, 512),        # Added additional layer
            nn.ReLU(),
            nn.Dropout(0.5),
            
            nn.Linear(512, 256),        # Decreased back to 256
            nn.ReLU(),
            nn.Dropout(0.5),
            
            nn.Linear(256, 128),        # Decreased back to 128
            nn.ReLU(),
            nn.Dropout(0.5),
            
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, X):
        return self.model(X)
