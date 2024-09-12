import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
import os

class SGR(nn.Module):
    def __init__(self, input_dim):
        super(SGR, self).__init__()
        self.model = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, 128),
            nn.LeakyReLU(),  # Changed activation to LeakyReLU
            nn.Dropout(0.5),

            nn.Linear(128, 256),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            
            nn.Linear(256, 512),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, X):
        return self.model(X)

