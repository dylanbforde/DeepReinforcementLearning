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
        
        # Initialize adaptive threshold with base value
        self.base_threshold = suitability_threshold
        self.suitability_threshold = suitability_threshold
        self.adjustment_factor = adjustment_factor
        self.device = device
        
        # Added to track episodes and performance
        self.episode_count = 0
        self.shift_history = []  # Store recent domain shift values
        self.recent_losses = []  # Track recent loss values
        self.history_size = 100  # How many recent values to consider
        self.update_count = 0    # Track number of model updates
        
        # Dynamic threshold adjustment parameters
        self.threshold_min = 0.2
        self.threshold_max = 0.8
        self.adapt_frequency = 50  # How often to adjust the threshold
    
    def adapt_threshold(self):
        """Adaptively adjust the suitability threshold based on recent performance."""
        if len(self.recent_losses) < 10:
            return  # Not enough data to make adjustments
            
        # Calculate loss trend (improving or worsening)
        if len(self.recent_losses) >= 20:
            recent_avg = sum(self.recent_losses[-10:]) / 10
            older_avg = sum(self.recent_losses[-20:-10]) / 10
            loss_improving = recent_avg < older_avg
            
            # If loss is improving, we can be more selective (higher threshold)
            # If loss is worsening, be more adaptive (lower threshold)
            if loss_improving:
                self.suitability_threshold = min(
                    self.threshold_max, 
                    self.suitability_threshold + 0.01
                )
            else:
                self.suitability_threshold = max(
                    self.threshold_min, 
                    self.suitability_threshold - 0.02
                )
        
        # Also consider recent domain shifts
        if len(self.shift_history) >= 10:
            avg_shift = sum(self.shift_history[-10:]) / 10
            # If encountering significant domain shifts, lower threshold to be more sensitive
            if avg_shift > 0.5:
                self.suitability_threshold = max(self.threshold_min, self.suitability_threshold - 0.05)

    def predict_suitability(self, state, domain_shift_metric):
        """Predict how suitable the current policy is for the current environment state."""
        # Track domain shift values for threshold adjustment
        self.shift_history.append(domain_shift_metric.item())
        if len(self.shift_history) > self.history_size:
            self.shift_history = self.shift_history[-self.history_size:]
            
        # Create input by concatenating state with domain shift metric
        predictor_input = torch.cat((state, domain_shift_metric.unsqueeze(1)), dim=1)
        predicted_suitability = self.model(predictor_input)
        
        # Periodically adjust the threshold
        if self.update_count % self.adapt_frequency == 0 and self.update_count > 0:
            self.adapt_threshold()
            
        return predicted_suitability

    def update(self, state, domain_shift_metric, true_suitability, random_suitability=None):
        """Update the domain shift predictor model using the provided data."""
        # Skip duplicate updates
        self.update_count += 1
        
        # Use appropriate training target
        if random_suitability is not None and self.episode_count < 200:
            # For initial training, use random values to explore
            suitability = random_suitability
        else:
            # Use model's prediction
            suitability = self.predict_suitability(state, domain_shift_metric)

        # Perform backpropagation
        self.optimizer.zero_grad()
        loss = self.loss_fn(suitability, true_suitability)
        loss.backward()
        
        # Apply gradient clipping to stabilize training
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        # Track loss history for adaptive threshold
        loss_val = loss.item()
        self.recent_losses.append(loss_val)
        if len(self.recent_losses) > self.history_size:
            self.recent_losses = self.recent_losses[-self.history_size:]
        
        # Increase episode count after each update call
        self.episode_count += 1
        return loss_val, suitability.detach()

class DomainShiftNN(nn.Module):
    """
    Enhanced Neural Network for assessing domain shift suitability.
    
    Features:
    - Deeper architecture with skip connections
    - Dropout for regularization
    - Layer normalization for training stability
    - More sophisticated activation functions
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        """
        Initializes the improved neural network architecture.

        Args:
            input_dim (int): The dimension of the input (state + domain shift value).
            hidden_dim (int): The dimension of the hidden layers.
            output_dim (int): The dimension of the output layer.
        """
        super(DomainShiftNN, self).__init__()
        
        # Feature extraction layers
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        torch.nn.init.kaiming_normal_(self.input_layer.weight, nonlinearity='leaky_relu')
        
        # Hidden layers with residual connections
        self.hidden1 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        torch.nn.init.kaiming_normal_(self.hidden1.weight, nonlinearity='leaky_relu')
        
        self.hidden2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln3 = nn.LayerNorm(hidden_dim)
        torch.nn.init.kaiming_normal_(self.hidden2.weight, nonlinearity='leaky_relu')
        
        # Output layer
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        torch.nn.init.xavier_uniform_(self.output_layer.weight)
        
        # Regularization
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        """
        Forward pass through the enhanced network with skip connections.
        
        Args:
            x (torch.Tensor): The input tensor containing state and domain shift data.
        
        Returns:
            torch.Tensor: Suitability score between 0-1.
        """
        # Input processing
        x = F.leaky_relu(self.ln1(self.input_layer(x)), negative_slope=0.1)
        x = self.dropout(x)
        
        # First residual block
        residual = x
        x = F.leaky_relu(self.ln2(self.hidden1(x)), negative_slope=0.1)
        x = self.dropout(x)
        x = residual + x  # Skip connection
        
        # Second residual block
        residual = x
        x = F.leaky_relu(self.ln3(self.hidden2(x)), negative_slope=0.1)
        x = residual + x  # Skip connection
        
        # Output processing
        x = torch.sigmoid(self.output_layer(x))
        return x