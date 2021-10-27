from math import exp
from torch import nn
import torch
import torch.functional as F
from torch.utils.data import Dataset
import numpy as np

class ExperienceDataset(Dataset):
    def __init__(self,experience,device):
        self.experience = experience
        self.device = device
        

    def __len__(self):
        return len(self.experience)

    def __getitem__(self, idx):
        
        boards, policies, values = self.experience[idx]

        boards = torch.tensor(np.array(boards),device=self.device).float()
        target_policies = torch.tensor(np.array(policies),device=self.device).float()
        target_values = torch.tensor(np.array(values),device=self.device).float()

        return boards, target_policies, target_values

class PolicyValueNetwork(nn.Module):
    
    def __init__(self, input_size, hidden_size, nb_actions) -> None:
        super(PolicyValueNetwork, self).__init__()
        

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        

        self.feature_Net = nn.Sequential(
            nn.Conv2d(1,32,3,1,padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32,32,3,1,padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(32,32,3,1,padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(32,32,3,1,padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(32,32,3,1,padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(32),
            nn.Flatten(),
            nn.Linear(1344, hidden_size),
            nn.LeakyReLU()).float()

        self.policy = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size,nb_actions),
            nn.LeakyReLU(),
            nn.Softmax(dim=-1)).float()

        self.value = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size,1),
            nn.Tanh()).float()
        
        if torch.cuda.is_available():
            self.feature_Net.cuda()
            self.policy.cuda()
            self.value.cuda()
    
    
    def forward(self, obs):
        """
        Run a tensor board through the model
        """

        feature = self.feature_Net(obs)
        policy = self.policy(feature)
        value = self.value(feature)

        
        return policy, value

    def forward_no_grad(self, obs):
        """
        Run a numpy board through the model without keeping gradients 
        """
        
        board = torch.tensor(obs,device=self.device).float().unsqueeze(0).unsqueeze(0)
        
        with torch.no_grad():
            feature = self.feature_Net(board)
            policy = self.policy(feature).squeeze()
            value = self.value(feature).squeeze()

        return policy.cpu().numpy(), value.cpu().numpy()