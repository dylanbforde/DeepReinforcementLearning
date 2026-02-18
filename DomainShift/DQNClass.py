import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim, domain_shift_input_dim, hidden_dim=128):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(state_dim + domain_shift_input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, action_dim)
    
    def forward(self, x, domain_shift):
        domain_shift = domain_shift.view(-1, 1)  # Reshape to [batch_size, 1]
        x = torch.cat((x, domain_shift), dim=1)
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return torch.tanh(self.layer3(x))  # Use tanh activation for continuous actions