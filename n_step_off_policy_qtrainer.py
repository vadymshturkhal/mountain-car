from functools import reduce
import operator
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from game_settings import CAR_ACTION_LENGTH, WEIGHT_DECAY, BATCH_SIZE


class NStepOffPolicyQTrainer:
    def __init__(self, model, lr, gamma, n_steps=0):
        self._alpha = lr
        self._gamma = gamma
        self._model = model
        self._optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
        self._criterion = nn.SmoothL1Loss()
        self._n_steps = n_steps

    def train_n_steps(self, states: list, actions: list, rewards: list, dones: int, last_index=0, epsilon=1):
        if len(states) < 2:
            return 0

        if len(states) < self._n_steps:
            current_n_steps = len(states)
        else:
            current_n_steps = self._n_steps

        if last_index == 0:
            last_index = len(states)

        states = torch.tensor(states, dtype=torch.float)
        actions = torch.tensor(actions, dtype=torch.long)

        # Rho
        ratio = self._importance_sampling_ratio(states, epsilon)

        for param_group in self._optimizer.param_groups:
            param_group['lr'] = self._alpha * ratio

        # G
        rewards_gamma_sum = self._calculate_rewards(rewards, last_index=last_index)

        if not dones[last_index - 1]:
            # G + y**n * max(Q(S_tau+n, a_tau+n))
            rewards_gamma_sum += self._gamma**current_n_steps * torch.max(self._model(states[last_index - 1]).detach())

        q_values = self._model(states[last_index - current_n_steps])

        target = q_values.clone()
        target[torch.argmax(actions[last_index - current_n_steps]).item()] = rewards_gamma_sum
    
        self._optimizer.zero_grad()
        loss = self._criterion(target, q_values)
        loss.backward()

        self._optimizer.step()
        return loss.item()

    def train_episode(self, states: list, actions: list, rewards: list, dones: int):
        if len(states) < BATCH_SIZE:
            start_index = 0
            last_index = len(states)
        else:
            start_index = random.randint(0, len(states) - BATCH_SIZE)
            last_index = start_index + BATCH_SIZE

        for relative_last_index in range(start_index + self._n_steps, last_index):
            self.train_n_steps(states, actions, rewards, dones, last_index=relative_last_index)

    def _calculate_rewards(self, rewards, last_index=None):
        rewards_gamma_sum = 0
        if last_index is None:
            last_index = len(rewards)
        start_index = last_index - self._n_steps

        for i in range(start_index, last_index):
            rewards_gamma_sum += rewards[i] * self._gamma**(i - start_index)
        return rewards_gamma_sum

    def _importance_sampling_ratio(self, states, epsilon, last_index=None):
        if last_index is None:
            last_index = len(states)
        start_index = last_index - self._n_steps
        last_index -= 1
        ratios = []

        with torch.no_grad():
            for i in range(start_index, last_index):
                logits = self._model(states[i])

                # Apply softmax to convert logits to probabilities
                probabilities = F.softmax(logits, dim=-1)
                action_index = torch.argmax(probabilities)
                pi = probabilities[action_index]

                # B
                b = epsilon * (1 / CAR_ACTION_LENGTH) + (1 - epsilon)

                ratios.append(pi/b)

        # Using reduce with operator.mul
        ratios_multiplied = reduce(operator.mul, ratios)
        return ratios_multiplied