import torch
import torch.nn as nn

class SGR(nn.Module):
    def __init__(self, input_dim, conv_out_channels=16, conv_kernel_size=3, pool_size=2):
        super(SGR, self).__init__()
        
        # Define the 1D Convolutional and Pooling layers
        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=conv_out_channels, kernel_size=conv_kernel_size),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=pool_size),
            nn.Dropout(0.5),
            nn.Conv1d(in_channels=conv_out_channels, out_channels=conv_out_channels*2, kernel_size=conv_kernel_size),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=pool_size),
            nn.Dropout(0.5)
        )
        
        # Define the fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(conv_out_channels*2*7, 100),  # Adjust the input dimension based on the output size of Conv1d layers
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(100, 200),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(200, 1),
            nn.Sigmoid()
        )

    def forward(self, X):
        # Assume X is of shape (batch_size, input_dim, sequence_length)
        X = self.conv_layers(X)
        X = X.view(X.size(0), -1)  # Flatten the output for the fully connected layers
        X = self.fc_layers(X)
        return X

