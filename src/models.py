# src/models.py

# DO NOT import torch here
# import torch.nn as nn → move this inside the model class

class GCNModel:
    def __init__(self, hidden_channels):
        import torch.nn as nn  # ✅ move here
        import torch           # ✅ move here

        self.nn = nn.Sequential(
            nn.Linear(10, hidden_channels),  # Example structure
            nn.ReLU(),
            nn.Linear(hidden_channels, 1)
        )

    def __call__(self, x):
        return self.nn(x)
