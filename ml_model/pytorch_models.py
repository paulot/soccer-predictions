import torch
import torch.nn as nn
import torch.nn.functional as F

class OutcomeNN(nn.Module):
    """
    Neural Network to predict pass outcome (Success vs Turnover).
    Outputs a single probability (turnover rate).
    """
    def __init__(self, input_dim):
        super(OutcomeNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        return torch.sigmoid(self.fc3(x))

class DestinationNN(nn.Module):
    """
    Neural Network to predict pass destination zone (0 to 29).
    Outputs a probability distribution over the 30 zones.
    """
    def __init__(self, input_dim, output_dim=30):
        super(DestinationNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_dim)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        # Note: We output raw logits. Softmax is applied in the loss function (CrossEntropyLoss)
        # and explicitly during inference.
        return self.fc3(x)
