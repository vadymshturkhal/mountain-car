import gymnasium as gym
import numpy as np
from game_settings import CAR_WEIGHTS_FILENAME

from n_step_off_policy_agent import NStepOffPolicyQLearning


class TrainAgent:
    def __init__(self, epochs, is_rendering):
        if is_rendering:
            render_mode = "human"
        else:
            render_mode = "rgb_array"

        self.env = gym.make('MountainCar-v0', render_mode=render_mode)
        self.car_agent =  NStepOffPolicyQLearning(*[is_load_weights, CAR_WEIGHTS_FILENAME, epochs, is_load_n_games])
        self.epochs = epochs
        self._car_game_reward = 0
        self._observations = []
        self._actions = []
        self._rewards = []
        self._dones = []
        self._losses = []
        self._max_x = float('-inf')

    def train(self):
        for _ in range(self.epochs):
            self._train_single_game()
            self._clear_game_data()

    def _train_single_game(self):
        max_reward = float('-inf')
        prev_observation = self.env.reset()[0]
        max_x = float('-inf')
        while True:
            action = self.car_agent.get_action(prev_observation)
            action = np.argmax(action)

            # Step through the environment and get five return values
            observation, reward, terminated, truncated, info = self.env.step(action)

            if prev_observation[0] > max_x:
                max_x = prev_observation[0]
                reward = 1

            self._observations.append(prev_observation)
            self._actions.append(action)
            self._rewards.append(reward)

            self._car_game_reward += reward
            self.car_agent.last_reward = reward

            prev_observation = observation
            done = terminated or truncated

            self._dones.append(float(done))
            loss = self.car_agent.train_step(self._observations, self._actions, self._rewards, self._dones)
            self._losses.append(loss)

            if done:
                print('done')
                self._dones.append(1)

                self.car_agent.n_games += 1

                # Save snake model
                if self._car_game_reward >= max_reward:
                    max_reward = self._car_game_reward
                    self.car_agent.model.save(epoch=self.car_agent.n_games, filename=CAR_WEIGHTS_FILENAME)

                average_loss = sum(self._losses) / len(self._losses)
                print(average_loss, self.car_agent.epsilon)
                break

    def _clear_game_data(self):
        self._car_game_reward = 0
        self._observations.clear()
        self._actions.clear()
        self._rewards.clear()
        self._dones.clear()
        self._losses.clear()


epochs = 100
# is_rendering = False
# is_load_weights = False
# is_load_n_games = False
is_rendering = True
is_load_weights = False
is_load_n_games = False
train_agent = TrainAgent(epochs, is_rendering)
train_agent.train()
