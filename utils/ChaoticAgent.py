import numpy as np
import random

class ChaoticAgent():

    def __init__(self, num_arms):
        self.num_arms = num_arms

    def choose_action_move(self):
        random.seed()
        mu = random.randint(0,2)
        return mu

    def select_arm(self):
        random.seed()
        zeta = random.randint(0, self.num_arms-1)
        return zeta

