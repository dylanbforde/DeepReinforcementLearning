import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class DomainShiftPredictor:
    def __init__(self, input_dim, hidden_dim, output_dim, lr, suitability_threshold, adjustment_factor, device):
        self.model = DomainShiftNN(input_dim, hidden_dim, output_dim).to(device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, amsgrad=True)
        self.loss_fn = nn.BCELoss()
        self.suitability_threshold = suitability_threshold
        self.adjustment_factor = adjustment_factor
        self.device = device
        # Added to keep track of episodes
        self.episode_count = 0

    def predict_suitability(self, state, domain_shift_metric):
        predictor_input = torch.cat((state, domain_shift_metric.unsqueeze(1)), dim=1)
        predicted_suitability = self.model(predictor_input)
        return predicted_suitability

    def update(self, state, domain_shift_metric, true_suitability, random_suitability=None, predicted_suitability=None):
        # If random_suitability is provided and we are in the first 200 episodes, use it for training
        if predicted_suitability is not None:
            suitability = predicted_suitability
        elif random_suitability is not None and self.episode_count < 200:
            suitability = random_suitability
        else:
            suitability = self.predict_suitability(state, domain_shift_metric)

        self.optimizer.zero_grad()
        loss = self.loss_fn(suitability, true_suitability)
        loss.backward()
        self.optimizer.step()
        # Increase episode count after each update call
        self.episode_count += 1
        return loss.item(), suitability.detach()

class DomainShiftNN(nn.Module):
    """
    Neural Network for assessing the domain shift suitability.
    
    Attributes:
        fc1 (torch.nn.Linear): First fully connected layer.
        fc2 (torch.nn.Linear): Second fully connected layer.
        fc3 (torch.nn.Linear): Third fully connected layer, output layer.
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        """
        Initializes the neural network layers.

        Args:
            input_dim (int): The dimension of the input (state + domain shift value).
            hidden_dim (int): The dimension of the hidden layers.
            output_dim (int): The dimension of the output layer.
        """
        super(DomainShiftNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        torch.nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='relu')
        self.fc1.bias.data.fill_(0.01)

        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        torch.nn.init.kaiming_normal_(self.fc2.weight, nonlinearity='relu')
        self.fc2.bias.data.fill_(0.01)

        self.fc3 = nn.Linear(hidden_dim, output_dim)
        torch.nn.init.kaiming_normal_(self.fc3.weight, nonlinearity='relu')
        self.fc3.bias.data.fill_(0.01)

    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): The input tensor.
        
        Returns:
            torch.Tensor: The output of the network.
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))  # Assuming binary classification (0 or 1) for suitability
        return x