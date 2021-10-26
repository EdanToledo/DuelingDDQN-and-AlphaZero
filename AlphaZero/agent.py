import random
from torch._C import device
from Connect4 import Connect4Game
from MCTS import MCTS
import numpy as np
import time
import torch
from model import ExperienceDataset, PolicyValueNetwork
from torch.optim import AdamW
import torch.nn.functional as F
from torch.utils.data import DataLoader

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)


class AlphaZero:

    def __init__(self, game, input_size, hidden_size, nb_actions, num_simulations, learning_rate=0.0001):
        self.game = game

        self.model = PolicyValueNetwork(input_size, hidden_size, nb_actions)
        self.mcts = MCTS(self.game, self.model, num_simulations)

        self.optimizer = AdamW(self.model.parameters(), lr=learning_rate)

        self.best_model = PolicyValueNetwork(
            input_size, hidden_size, nb_actions)
        self.best_model.load_state_dict(self.model.state_dict())
        self.best_mcts = MCTS(self.game, self.best_model, num_simulations)

    def act(self, board, player, best=False, temperature=0,noise_weight=0):
        """
        choose an action according to mcts algorithm - choose to use current or best weights
        """
        current_player_board = self.game.get_perspective_board(board, player)
        
       
        if best:
            root = self.best_mcts.run(current_player_board, to_play=1,noise_weight=noise_weight)
        else:
            root = self.mcts.run(current_player_board, to_play=1,noise_weight=noise_weight)

        action = root.select_action(temperature=temperature)

        return action

    def get_value(self,board,player):
        """
        get value from value network
        """
        current_player_board = self.game.get_perspective_board(board,player)

        _ , value = self.model.forward_no_grad(current_player_board)
    
        return value

    def generate_data_self_play(self, episodes, temperature=1,noise_weight=0.75):
        """
        Play games against itself and generate tuples of experience to train policy and value network
        """
        experience = []
        for episode in range(episodes):
            train_examples = []
            current_player = 1
            state = self.game.get_init_board()

            while True:
                current_player_board = self.game.get_perspective_board(
                    state, current_player)

                root = self.mcts.run(current_player_board, to_play=1,noise_weight=noise_weight)

                action_probs = self.mcts.get_improved_action_probabilities(
                    root)

                train_examples.append(
                    (current_player_board, current_player, action_probs))

                action = root.select_action(temperature=temperature)

                state, current_player = self.game.get_next_state(
                    state, current_player, action)

                reward = self.game.get_reward_for_player(state, current_player)

                if reward is not None:

                    for hist_state, hist_current_player, hist_action_probs in train_examples:

                        experience.append(
                            (hist_state, hist_action_probs, reward * ((-1) ** (hist_current_player != current_player))))

                    break

        return experience

    def train(self, episodes, epochs, batch_size, model_path, iterations, arena_games):
        """
        Train the agent
        """

        self.model.train()

        # Number of training iterations - each iteration generates new self-play data using updated model
        for iter in range(iterations):
            print("Iteration", iter, ":")
            # Generate experience
            experience = (self.generate_data_self_play(episodes))
            training_data = ExperienceDataset(experience, self.model.device)

            train_dataloader = DataLoader(
                training_data, batch_size=batch_size, shuffle=True)

            for epoch in range(epochs):
                epoch_loss = 0
                for boards, target_policies, target_values in train_dataloader:

                    predicted_policies, predicted_values = self.model(
                        boards.unsqueeze(1))

                    policy_loss = - \
                        (target_policies*torch.log(predicted_policies+1e-7)).sum(-1).mean()

                    value_loss = F.mse_loss(
                        target_values.squeeze(), predicted_values.squeeze())

                    total_loss = policy_loss + value_loss

                    self.optimizer.zero_grad()

                    total_loss.backward()

                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 2.0)

                    self.optimizer.step()

                    epoch_loss += total_loss.detach()

                print("Total Epoch Loss:", epoch_loss)

            self.arena(arena_games)
            self.model.load_state_dict(self.best_model.state_dict())
            torch.save(self.model.state_dict(), model_path)

    def load_weights(self, path):
        """
        Load external weights
        """
        self.model.load_state_dict(torch.load(path))
        self.best_model.load_state_dict(torch.load(path))

    def set_new_best(self):
        """
        Set the best model to the current models parameters.
        """
        self.best_model.load_state_dict(self.model.state_dict())

    def arena(self, no_games=100):
        """
        plays no_games games using current model against best model - if current model wins rougly 55% of the games it becomes the new best model
        """
        wins = []
        for i in range(no_games):
            board = self.game.get_init_board()

            game_over = False
            turn = 0

            while not game_over:
                if turn == 0:
                    col = self.act(board, 1, False,1,0.25)

                    if self.game.is_valid_location(board, col):
                        row = self.game.get_next_open_row(board, col)
                        self.game.drop_piece(board, row, col, 1)

                    reward = self.game.get_reward_for_player(board, 1)

                    if reward is not None:
                        wins.append(1 if reward == 1 else 0)
                        game_over = True

                else:

                    col = self.act(board, -1, True,1,0.25)
                    if self.game.is_valid_location(board, col):
                        row = self.game.get_next_open_row(board, col)
                        self.game.drop_piece(board, row, col, -1)

                    reward = self.game.get_reward_for_player(board, -1)

                    if reward is not None:
                        wins.append(0 if reward == 1 or reward == 0 else 1)
                        game_over = True

                turn += 1
                turn = turn % 2

        if sum(wins) > int(no_games/1.8):
            self.set_new_best()
        print("Arena Win Percentage:", 100*sum(wins)/no_games)
