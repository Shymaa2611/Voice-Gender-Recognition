import torch
import torch.nn as nn

class SGR(nn.Module):
    def __init__(self, input_dim, conv_out_channels=16, conv_kernel_size=3, pool_size=2):
        super(SGR, self).__init__()
        
        # Convolutional layers for audio feature extraction
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
        
        # Determine the output size after convolutional layers
        self._to_linear = None
        self.convs_output_size(input_dim)
        
        self.fc_layers = nn.Sequential(
            nn.Linear(self._to_linear, 100),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(100, 200),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(200, 1),
            nn.Sigmoid()
        )

    def convs_output_size(self, input_dim):
        # Create a dummy input to determine the size after conv layers
        dummy_input = torch.ones(1, input_dim, 768)  # Adjust sequence length as needed
        dummy_output = self.conv_layers(dummy_input)
        self._to_linear = dummy_output.numel()

    def forward(self, X):
        X = self.conv_layers(X)
        X = X.view(X.size(0), -1)  # Flatten the output for the fully connected layers
        X = self.fc_layers(X)
        return X
