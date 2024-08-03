import numpy as np
import random

class GAgent:
    def __init__(self, num_tasks, sval, start_task, exploration_rate=0.1):
        # Player values
        self.threshold = sval
        self.current_task = start_task
        self.exploration_rate = exploration_rate

        # Player data storage
        self.sum_table = np.zeros((num_tasks))
        self.count_table = np.zeros((num_tasks))
        self.satisfaction_table = np.zeros((num_tasks))  # assume all coalitions meet threshold initially so they explore
        self.num_tasks = num_tasks

    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def choose_action_move(self):
        """Choose a new action based on the satisfaction of neighboring tasks."""

        left_task = (self.current_task - 1) % self.num_tasks
        right_task = (self.current_task + 1) % self.num_tasks

        current_satisf = self.satisfaction_table[self.current_task]
        left_satisf = self.satisfaction_table[left_task]
        right_satisf = self.satisfaction_table[right_task]

        if np.random.rand() < self.exploration_rate:
            # Explore with a small probability
            mu = random.randint(1, 2)
        else:
            # Select the move that leads to the highest satisfaction
            satisfactions = [current_satisf, left_satisf, right_satisf]
            mu = np.argmax(satisfactions)

        if mu == 1:
            self.current_task = left_task
        elif mu == 2:
            self.current_task = right_task

        return mu

    def update(self, reward):
        self.count_table[self.current_task] += 1
        self.sum_table[self.current_task] += reward

        self.satisfaction_table[self.current_task] = self.sum_table[self.current_task] / \
            self.count_table[self.current_task]