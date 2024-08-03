import numpy as np
import random

class GAgent():

    def __init__(self, num_tasks, num_arms):
        self.move_table = np.zeros(3)
        self.comm_table = np.zeros(num_arms)

    def choose_action_move(self):
        random.seed()
        mu = random.randint(0,2)
        return mu

    def select_arm(self):
        random.seed()
        zeta = random.randint(0, self.num_arms-1)
        return zeta

    def update(self, chosen_move, chosen_comm):
        pass