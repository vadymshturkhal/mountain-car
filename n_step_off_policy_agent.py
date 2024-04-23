import torch
import numpy as np

from model import Linear_QNet
from n_step_off_policy_qtrainer import NStepOffPolicyQTrainer

from game_settings import LR, CAR_ACTION_LENGTH, TRAINER_STEPS
from game_settings import CAR_INPUT_LAYER_SIZE, CAR_HIDDEN_LAYER_SIZE1, CAR_HIDDEN_LAYER_SIZE2, CAR_OUTPUT_LAYER_SIZE
from game_settings import CAR_GAMMA, CAR_MIN_EPSILON, CAR_START_EPSILON

class NStepOffPolicyQLearning:
    def __init__(self, is_load_weights=False, weights_filename=None, epochs=100, is_load_n_games=True, n_steps=TRAINER_STEPS):
        self.epsilon = 1
        self.epochs = epochs

        self.gamma = CAR_GAMMA
        self.n_steps = n_steps

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model = Linear_QNet(CAR_INPUT_LAYER_SIZE, CAR_HIDDEN_LAYER_SIZE1, CAR_HIDDEN_LAYER_SIZE2, CAR_OUTPUT_LAYER_SIZE)
        self.model.to(self.device)
        self.last_action = [0] * CAR_ACTION_LENGTH

        # Load the weights onto the CPU or GPU
        if is_load_weights:
            checkpoint = torch.load(weights_filename, map_location=self.device)
            self.n_games = checkpoint['epoch'] if is_load_n_games else 0
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
        else:
            self.n_games = 0

        self.trainer = NStepOffPolicyQTrainer(self.model, lr=LR, gamma=self.gamma, n_steps=n_steps)
    
    def train_step(self, states: list, actions: list, rewards: list, dones: list, last_index=0):
        loss = self.trainer.train_n_steps(states, actions, rewards, dones, last_index=last_index, epsilon=self.epsilon)
        return loss
    
    def train_n_steps(self, states: list, actions: list, rewards: list, dones: list, last_index=0) -> float:
        loss = self.trainer.train_n_steps(states, actions, rewards, dones, last_index=last_index, epsilon=self.epsilon)
        return loss

    # Update the estimates of action values
    def train_episode(self, states: list, actions: list, rewards: list, dones: list):
        self.trainer.train_episode(states, actions, rewards, dones)

    def get_action(self, state, is_train=True) -> np.array:
        """
        Return vector [0,0,0] with first two positions represent rotation, third pushes forward.
        Adjusted for epsilon-soft policy.
        """
        # Linear decay rate
        if is_train:
            self._update_epsilon_linear()
        else:
            self.epsilon = CAR_MIN_EPSILON

        final_move = [0] * CAR_ACTION_LENGTH
        if np.random.rand() < self.epsilon:
            # Distribute a small probability to all actions, keeping the majority for the best action
            probabilities = np.ones(CAR_ACTION_LENGTH) * (self.epsilon / CAR_ACTION_LENGTH)

            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            best_move = torch.argmax(prediction).item()

            # Adjust probability for the best action
            probabilities[best_move] += (1.0 - self.epsilon)

            # Choose action based on modified probabilities
            move = np.random.choice(np.arange(CAR_ACTION_LENGTH), p=probabilities)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        self.last_action = final_move
        return np.array(final_move)

    def _update_epsilon_linear(self):
        """
        Update the epsilon value linearly from start_epsilon to end_epsilon over total_games.
        
        Parameters:
        - current_game: The current game number (1 to total_games).
        - start_epsilon: The starting value of epsilon at game 1.
        - end_epsilon: The final value of epsilon at game total_games.
        - total_games: The total number of games over which epsilon will decay.
        
        Returns:
        - Updated epsilon value for the current game.
        """

        # Calculate the amount of decay per game
        decay_per_game = (CAR_START_EPSILON - CAR_MIN_EPSILON) / (self.epochs)
        
        # Update epsilon linearly based on the current game number
        new_epsilon = CAR_START_EPSILON - (decay_per_game * (self.n_games))
        
        # Ensure epsilon does not go below the end_epsilon
        self.epsilon = new_epsilon
        self.epsilon = max(self.epsilon, CAR_MIN_EPSILON)
