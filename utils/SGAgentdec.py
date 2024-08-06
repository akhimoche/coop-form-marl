import numpy as np
import random

class GAgent:
    def __init__(self, num_tasks, sval, start_task, initial_temperature=1.0, temperature_decay=0.92, min_temperature=0.1):
        # Player values
        self.threshold = sval
        self.current_task = start_task
        self.temperature = initial_temperature
        self.initial_temperature = initial_temperature
        self.temperature_decay = temperature_decay
        self.min_temperature = min_temperature

        # Player data storage
        self.sum_table = np.zeros((num_tasks))
        self.count_table = np.zeros((num_tasks))
        self.satisfaction_table = np.zeros((num_tasks))  # assume all coalitions meet threshold initially so they explore
        self.num_tasks = num_tasks

    def softmax(self, x, temperature):
        e_x = np.exp((x - np.max(x)) / temperature)
        return e_x / e_x.sum()

    def choose_action_move(self):
        """Choose a new action based on the satisfaction of neighboring tasks using Boltzmann exploration."""

        left_task = (self.current_task - 1) % self.num_tasks
        right_task = (self.current_task + 1) % self.num_tasks

        current_satisf = self.satisfaction_table[self.current_task]
        left_satisf = self.satisfaction_table[left_task]
        right_satisf = self.satisfaction_table[right_task]

        # Use Boltzmann exploration to probabilistically choose a move
        satisfactions = np.array([current_satisf, left_satisf, right_satisf])
        probabilities = self.softmax(satisfactions, self.temperature)

        mu = np.random.choice([0, 1, 2], p=probabilities)

        if mu == 1:
            self.current_task = left_task
        elif mu == 2:
            self.current_task = right_task

        # Decay the temperature
        self.temperature = max(self.min_temperature, self.temperature * self.temperature_decay)

        return mu

    def update(self, reward):
        self.count_table[self.current_task] += 1
        self.sum_table[self.current_task] += reward

        self.satisfaction_table[self.current_task] = self.sum_table[self.current_task] / \
            self.count_table[self.current_task]