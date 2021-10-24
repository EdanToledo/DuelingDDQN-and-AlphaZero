from torch import nn
import torch.nn as nn
import torch.nn.functional as F
import torch


class DQN(nn.Module):
    
    

    def __init__(self, input_size,channels, hidden_size, nb_actions) -> None:
        super(DQN, self).__init__()
        
        w=input_size[1]
        h=input_size[2]
        
        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        
                
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)),3,3)
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)),3,3)
        
        linear_input_size = convw * convh * 32
        

        self.DQN = nn.Sequential(
            nn.Conv2d(channels, 16, kernel_size=5, stride=2),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            nn.Conv2d(16, 32, kernel_size=5, stride=2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.MaxPool2d(3),
            nn.Flatten(),
            nn.Linear(linear_input_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, nb_actions))
        
    

    def forward(self, obs):

        return self.DQN(obs)

class Dueling_DQN(nn.Module):

    def __init__(self, input_size,channels, hidden_size, nb_actions) -> None:
        
        
        super(Dueling_DQN, self).__init__()

        w=input_size[1]
        h=input_size[2]

        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        
                
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)),3,3)
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)),3,3)
        
        linear_input_size = convw * convh * 32


        self.feature_layer = nn.Sequential(
            nn.Conv2d(channels, 16, kernel_size=5, stride=2),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            nn.Conv2d(16, 32, kernel_size=5, stride=2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.MaxPool2d(3),
            nn.Flatten())
        
        self.value_function = nn.Sequential(
                            nn.Linear(linear_input_size, hidden_size),
                            nn.LeakyReLU(),
                            nn.Linear(hidden_size, nb_actions))

        self.advantage_function = nn.Sequential(
                            nn.Linear(linear_input_size, hidden_size),
                            nn.LeakyReLU(),
                            nn.Linear(hidden_size, nb_actions))
        
    def forward(self,obs):
        feature_state = self.feature_layer(obs)
        state_value = self.value_function(feature_state)
        advantage_values = self.advantage_function(feature_state)
        q_values = state_value + (advantage_values - advantage_values.mean())
        
        return q_values