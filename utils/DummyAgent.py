import numpy as np

class DummyAgent():

    def __init__(self, num_arms):
        random.seed()
        self.chosen_zeta = random.randint(0, num_arms)

    def choose_action_move(self):
        mu = 0
        return mu

    def select_arm(self):
        zeta = self.chosen_zeta
        return zeta

