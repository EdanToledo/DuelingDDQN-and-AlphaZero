from torch.distributions import Categorical, Normal
from torch.optim import AdamW
import torch
from utils import Replay_Memory, Transition
from model import DQN, Dueling_DQN
import numpy as np
import random


class DQN_Agent:

    def __init__(self, input_size, channels, hidden_size, nb_actions, discount_factor=0.9, learning_rate=0.0001,
                 epsilon_start=0.9, epsilon_end=0.05, epsilon_anneal_over_steps=10000, batch_size=256,
                 update_frequency=1000, replay_memory_size=100000, use_dueling=True, use_DDQN=True):

        self.use_DDQN=use_DDQN
        
        if use_dueling:

            self.online_net = Dueling_DQN(
                input_size, channels, hidden_size, nb_actions)

            self.target_net = Dueling_DQN(
                input_size, channels, hidden_size, nb_actions)

        else:
            # Create the online network that the agent uses to select actions
            self.online_net = DQN(input_size, channels,
                                  hidden_size, nb_actions)

            # Create the target network that the agent uses in it's updates
            self.target_net = DQN(input_size, channels,
                                  hidden_size, nb_actions)

        if torch.cuda.is_available():
            self.device = torch.device("cuda")

            self.online_net.cuda()
            self.target_net.cuda()

        else:
            self.device = torch.device("cpu")

        self.num_actions = nb_actions

        # Set the target net's initial weights to equal the online net
        self.target_net.load_state_dict(self.online_net.state_dict())

        self.gamma = discount_factor

        # User Huber Loss function - others can be use such as MSE
        self.loss_function = torch.nn.SmoothL1Loss()

        # DQN optimizer
        self.optimizer = AdamW(self.online_net.parameters(), learning_rate)

        # Set initial and final epsilon values for epsilon greedy and choose the number of environment steps to anneal over
        self.epsilon_end = epsilon_end
        self.epsilon_start = epsilon_start
        self.epsilon_anneal_over_steps = epsilon_anneal_over_steps
        self.step_no = 0

        # Create Replay Memory and assign batch_size
        self.replay = Replay_Memory(replay_memory_size)
        self.batch_size = batch_size

        # Set update Frequency
        self.update_frequency = update_frequency

        # Best Weights for Eval
        self.best_weights = self.online_net.state_dict()

    def get_epsilon(self):
        """
        Get current epsilon value according to agents total step number in the environment
        """
        eps = self.epsilon_end
        if self.step_no < self.epsilon_anneal_over_steps:
            eps = self.epsilon_start - self.step_no * \
                ((self.epsilon_start - self.epsilon_end) /
                 self.epsilon_anneal_over_steps)

        return eps

    def act(self, obs):
        # Increment global step count
        self.step_no += 1

        if np.random.uniform() > self.get_epsilon():
            # Dont store gradients when acting in the environment
            with torch.no_grad():
                return torch.argmax(self.online_net(obs), dim=-1).view(1)
        else:

            return torch.tensor([random.randrange(self.num_actions)], dtype=torch.long, device=self.device)

    # store experience in replay memory
    def cache(self, state, action, reward, next_state, done):
        self.replay.push(state, action, reward, next_state, done)

    def train(self):
        self.online_net.train()
        self.target_net.train()
        self.target_net.load_state_dict(self.online_net.state_dict())

    def eval(self):
        self.online_net.eval()
        self.target_net.eval()

    def update_model(self):

        # Get minibatch of data from experience buffer
        batch = self.replay.sample(self.batch_size)

        # If memory doesnt have enough transitions
        if batch == None:
            return

        # Format batch to get a tensor of states, actions, rewards, next states and done booleans
        batch_tuple = Transition(*zip(*batch))

        state = torch.cat(batch_tuple.state)

        action = torch.stack(batch_tuple.action)

        reward = torch.stack(batch_tuple.reward)

        next_state = torch.cat(batch_tuple.next_state)

        done = torch.stack(batch_tuple.done)

        self.optimizer.zero_grad()

        # Get the Q values of the online nets current state and actions
        Q_Values = self.online_net(state).gather(1, action).squeeze()
        if self.use_DDQN:
            Q_Actions = self.online_net(next_state).max(dim=-1)[1].detach()
            # And choose the found action in the target net
            Q_Targets = reward + (1 - done.float()) * self.gamma * self.target_net(
                next_state).gather(1, Q_Actions.unsqueeze(1)).squeeze().detach()
        else:
            Q_Targets = reward + (1 - done.float()) * self.gamma * \
                self.target_net(next_state).max(dim=-1)[0].detach()

        # Calculate loss
        loss = self.loss_function(Q_Values, Q_Targets)

        # Calculate the gradients
        loss.backward()

        # Perform optimization step
        self.optimizer.step()

        # return loss.item()

    def update_target(self):
        """
        Update the target nets weights
        """
        self.target_net.load_state_dict(self.online_net.state_dict())

    def update_best_weights(self):
        self.best_weights = self.online_net.state_dict()

    def save_weights(self, path):
        torch.save(self.best_weights, path)

    def load_weights(self, path):
        self.online_net.load_state_dict(torch.load(path))
