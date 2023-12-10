import numpy as np
import random


class Agent2:

    def __init__(self, num_move, num_comm):
        self.num_move = num_move
        self.num_comm = num_comm
        self.num_arms = num_move * num_comm
        self.counts = np.ones((self.num_arms), dtype=float)  # Number of times each arm has been pulled
        self.values = np.zeros((self.num_arms), dtype=float)  # Estimated values of each arm



    def select_arm(self):
        # Select arm with the maximum UCB value
        ucb_values = self.values + np.sqrt(2 * np.log(sum(self.counts)) / (self.counts+1e-5 ))

        ucb_values = np.reshape(ucb_values, (self.num_move, self.num_comm))
        chosen_arm = np.argmax(ucb_values)

        if sum(self.values) == 0:
            chosen_arm = random.randint(0,self.num_arms-1)

        a_comm = chosen_arm % self.num_comm
        a_move = (chosen_arm - a_comm) / self.num_comm

        return int(a_move), int(a_comm)

    def update(self, arm, reward):
        # Update the estimates after pulling the selected arm and observing the reward
        self.counts[arm] += 1
        self.values[arm] += (reward - self.values[arm]) / self.counts[arm]