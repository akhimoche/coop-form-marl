import numpy as np
import random


class Agent2:

    def __init__(self, num_arms):
        self.num_arms = num_arms
        self.counts = np.ones((num_arms), dtype=float)  # Number of times each arm has been pulled
        self.values = np.zeros((num_arms), dtype=float)  # Estimated values of each arm

    def select_arm(self):
        # Select arm with the maximum UCB value
        ucb_values = self.values + np.sqrt(2 * np.log(sum(self.counts)) / (self.counts+1e-5 ))
        chosen_arm = np.argmax(ucb_values)
        return chosen_arm, 0

    def update(self, arm, reward):
        # Update the estimates after pulling the selected arm and observing the reward
        self.counts[arm] += 1
        self.values[arm] += (reward - self.values[arm]) / self.counts[arm]