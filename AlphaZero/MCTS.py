import math
import numpy as np
import torch

# With help from https://github.com/JoshVarty/AlphaZeroSimple/blob/master/monte_carlo_tree_search.py

def ucb_score(parent, child):
    """
    Upper confidence bound score
    """
    prior_score = child.prior * math.sqrt(parent.visit_count) / (child.visit_count + 1)
    if child.visit_count > 0:
        # The value of the child is from the perspective of the opposing player
        value_score = -child.value()
    else:
        value_score = 0

    return value_score + prior_score


class Node:
    def __init__(self, prior, to_play):
        self.visit_count = 0
        self.to_play = to_play
        self.prior = prior
        self.value_sum = 0
        self.children = {}
        self.state = None

    def expanded(self):
        """
        Checks to see if node has children i.e has been expanded
        """
        return len(self.children) > 0

    def value(self):
        """
        Calculates value of node
        """
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

    def select_action(self, temperature):
        """
        Select action according to the visit count distribution and the temperature.
        """
        visit_counts = np.array([child.visit_count for child in self.children.values()])
        actions = [action for action in self.children.keys()]
        if temperature == 0:
            action = actions[np.argmax(visit_counts)]
        elif temperature == float("inf"):
            action = np.random.choice(actions)
        else:
            # See paper appendix Data Generation
            visit_count_distribution = visit_counts ** (1 / temperature)
            visit_count_distribution = visit_count_distribution / sum(visit_count_distribution)
            action = np.random.choice(actions, p=visit_count_distribution)

        return action

    def select_child(self):
        """
        Select the child with the highest UCB score.
        """
        best_score = -np.inf
        best_action = -1
        best_child = None

        for action, child in self.children.items():
            score = ucb_score(self, child)
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child

        return best_action, best_child

    def expand(self, state, to_play, action_probs):
        """
        expand node and keep track of the prior policy probability given by neural network
        """
        self.to_play = to_play
        self.state = state
        for a, prob in enumerate(action_probs):
            if prob != 0:
                self.children[a] = Node(prior=prob, to_play=self.to_play * -1)


class MCTS:

    def __init__(self, game, model,simulation_count):
        self.game = game
        self.model = model
        self.simulation_count = simulation_count

    def get_improved_action_probabilities(self,node):
        """
        Get action probability distribution according to visit count of node children
        """
        action_probs = [0 for _ in range(self.game.get_action_size())]
        for action, child in node.children.items():
            action_probs[action] = child.visit_count

        action_probs = action_probs / np.sum(action_probs)
        
        return action_probs

    def add_dirichlet_noise(self,action_probs,weight):
       
        new_action_probs = (1-weight)*action_probs + (weight)*np.random.dirichlet(np.zeros([len(action_probs)])+2)
        
        return new_action_probs

    def run(self, state, to_play, noise_weight):
        """
        Run the MCTS algorithm and build search tree with simulations
        """

        root = Node(0, to_play)
        
        # EXPAND root
        action_probs, value = self.model.forward_no_grad(state)
        action_probs = self.add_dirichlet_noise(action_probs,noise_weight)
        valid_moves = self.game.get_valid_moves(state)
        action_probs = action_probs * valid_moves  # mask invalid moves
        
        if valid_moves == [0,0,0,0,0,0,0]:
            print("ERROR")
        
        action_probs /= np.sum(action_probs)
        root.expand(state, to_play, action_probs)

        for _ in range(self.simulation_count):
            node = root
            search_path = [node]

            # SELECT
            while node.expanded():
                action, node = node.select_child()
                search_path.append(node)

            parent = search_path[-2]
            state = parent.state
            # Now we're at a leaf node and we would like to expand
            # Players always play from their own perspective
            next_state, _ = self.game.get_next_state(state, player=1, action=action)
            # Get the board from the perspective of the other player
            next_state = self.game.get_perspective_board(next_state, player=-1)

            # The value of the new state from the perspective of the other player
            value = self.game.get_reward_for_player(next_state, player=1)
            if value is None:
                # If the game has not ended:
                # EXPAND
                action_probs, value = self.model.forward_no_grad(next_state)
                action_probs = self.add_dirichlet_noise(action_probs,noise_weight)
                valid_moves = self.game.get_valid_moves(next_state)
                action_probs = action_probs * valid_moves  # mask invalid moves
                action_probs /= np.sum(action_probs)
                node.expand(next_state, parent.to_play * -1, action_probs)

            self.backpropagate(search_path, value, parent.to_play * -1)

        return root

    def backpropagate(self, search_path, value, to_play):
        """
        At the end of a simulation, we propagate the evaluation all the way up the tree
        to the root.
        """
        for node in reversed(search_path):
            node.value_sum += value if node.to_play == to_play else -value
            node.visit_count += 1