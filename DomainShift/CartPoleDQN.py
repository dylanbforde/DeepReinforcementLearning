import torch
import torch.nn as nn
import torch.nn.functional as F

class CartPoleDQN(nn.Module):
    """
    Deep Q-Network for CartPole with discrete actions.
    """
    def __init__(self, state_dim, action_dim, domain_shift_input_dim):
        super(CartPoleDQN, self).__init__()
        self.layer1 = nn.Linear(state_dim + domain_shift_input_dim, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, action_dim)
        
        # Initialize weights with xavier/glorot initialization
        nn.init.xavier_uniform_(self.layer1.weight)
        nn.init.xavier_uniform_(self.layer2.weight)
        nn.init.xavier_uniform_(self.layer3.weight)
        
        # Initialize biases with small positive values to ensure initial activations
        self.layer1.bias.data.fill_(0.01)
        self.layer2.bias.data.fill_(0.01)
        self.layer3.bias.data.fill_(0.01)
    
    def forward(self, x, domain_shift):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): State tensor.
            domain_shift (torch.Tensor): Domain shift metric tensor.
            
        Returns:
            torch.Tensor: Q-values for each action.
        """
        domain_shift = domain_shift.view(-1, 1)  # Reshape to [batch_size, 1]
        x = torch.cat((x, domain_shift), dim=1)
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        # No activation for discrete action Q-values
        return self.layer3(x)