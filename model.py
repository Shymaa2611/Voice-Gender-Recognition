import torch.nn as nn

class SGR(nn.Module):
    def __init__(self, input_dim):
        super(SGR, self).__init__()
        self.model = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, 100),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(100, 200),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(200, 1),
            nn.Sigmoid()
        )

    def forward(self, X):
        # Flatten the input to match the expected input dimension
        X = X.view(X.size(0), -1)  # Flatten (batch_size, time_steps, n_mfcc) to (batch_size, input_dim)
        return self.model(X)
