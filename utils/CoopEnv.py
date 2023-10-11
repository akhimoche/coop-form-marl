#!/usr/bin/env pythonomm
# coding: utf-8

# In[1]:

import random
import gym
import numpy as np

class CoopEnv(gym.Env):

    # -------------------- Game Support Methods -------------------- # //
    def characteristic_function(self, C, sin_vals, task, n, a=None):

        """ Return the characteristic value of a coalition.

        """

        random.seed(a) # set seed
        bias = random.uniform(0, len(C)) # get a random mult. bias

        if len(C) == 1: # ... but individual coalitions are always have m.b = 1
            bias = 1

        # get sum of singleton values in coaliton ...
        singleton_value_sum_in_C = sum(sin_vals[f'Player {player}'] for player in C)
        # ... and multiply this sum by bias
        char_val = singleton_value_sum_in_C * bias

        return char_val


    def get_observations_from_CS(self, CS, locs, n):

        """ Given a coalition structure, return ordered list of binary strings representing
            set observations for each agent (i.e what set the agent is in).

        """

        observations_in_bit = []
        for player in range(n):

            bit_observation = np.zeros((n))
            C = CS[locs[f'Player {player+1}']]
            agents_within_C = [int(a)-1 for a in list(C)] # indices are 1 less than the player tag

            bit_observation[agents_within_C] = 1
            observations_in_bit.append(bit_observation)

        return observations_in_bit
    # // -------------------- Game Support Methods -------------------- #



    # -------------------- Game Phase Execution Methods -------------------- # //
    def movement_phase(self, CS, locs, n, actions):

        for player in range(n): # action will be index of task to join

            new_C = actions[player] # get new coalition index
            current_C = locs[f'Player {player + 1}'] # old coalition index

            CS[current_C].remove(f'{player + 1}') # remove from old coalition...
            CS[new_C].add(f'{player + 1}') # add player to new coalition...


            locs[f'Player {player + 1}'] = new_C # ... update locs


        next_state = self.get_observations_from_CS(CS, locs, n)
        return next_state

    def communication_phase(self, CS, sin_vals, n, cnf, a=None):

        comm_vals = {} # temp. dict. to store communicated singleton values
        random.seed(a) # reset the seed for random communication noise (temporary)
        for player in range(n):

            sin_val = sin_vals[f'Player {player + 1}']
            noise = random.uniform(-cnf, cnf) # communicated singleton value is original + noise
            comm_vals[f'Player {player + 1}'] = sin_val * (1 + noise)

        # Get payoffs for coalitions from char func:
        char_vals = [] # array to store char. values of each coalition in CS
        sum_vals = [] # array to store sum of comm. singleton values for each C in CS
        task = 0
        for C in CS:

            # get value for each coalition in coalition structure
            if len(C) > 0:

                seed = task*n + len(C)
                char_value = self.characteristic_function(C, sin_vals, task, n, seed)
                char_vals.append(char_value)

                sum_value = sum(comm_vals[f'Player {player}'] for player in C)
                sum_vals.append(sum_value)

            else:

                char_vals.append(0)
                sum_vals.append(0)

            task += 1

        return comm_vals, char_vals, sum_vals

    def payoff_dist_phase(self, comm_vals, char_vals, sum_vals, CS, locs, sin_vals, n):

        rewards = np.zeros((n))
        for player in range(n): # determine how payoff be divided for players in coalitions

            # global values #
            loc = locs[f'Player {player + 1}'] # player location (task index in CS)
            sin_val = sin_vals[f'Player {player + 1}'] # player singleton value


            # local values #
            comm_val = comm_vals[f'Player {player + 1}'] # communicated s. value
            char_val = char_vals[loc] # value of the coalition the player is in
            comm_sum = sum_vals[loc] # sum of singleton values in coalition

            payoff = (comm_val/comm_sum) * char_val
            rewards[player] = payoff

            # check stability with individual rationality
            if payoff < sin_val:

                # if an agent doesn't like coalition
                # all agents get 0 reward
                C = CS[loc]
                agents_within_C = [int(a)-1 for a in C]
                rewards[agents_within_C] = 0

        return rewards
    # // -------------------- Game Phase Execution Methods -------------------- #



    # -------------------- Gym Methods -------------------- # //
    def __init__(self, n, tasks, cnf=0.1):


        # Numerical Parameters:
        # ------------------------------------------------------ #
        self.n = n
        self.num_of_tasks = tasks
        self.done = False
        self.cnf = cnf # communication noise factor of s value
        # ------------------------------------------------------ #

        # Data Arrays:
        # ------------------------------------------------------ #
        self.CS = [set() for i in range(1, self.num_of_tasks + 1)] # starting coalition structure
        self.locs = {} #  location of each agent in the coalition structure
        self.sin_vals = {} # assigned singleton value to each agent
        # ------------------------------------------------------ #

        # Initialisation:
        # ------------------------------------------------------ #
        random.seed() # reset the seed
        for player in range(self.n): # choose a random task for each agent

            des_C = random.randint(0, self.num_of_tasks-1) # pick a random coalition...
            self.CS[des_C].add(f'{player +1 }') # ... and add the player to it...

            self.locs[f'Player {player + 1}'] = des_C # ... record index...
            self.sin_vals[f'Player {player + 1}'] = random.random() # ...and its value
        # ------------------------------------------------------ #


    def step(self, actions):

        """ Three phases:
            Movement - Agents' chosen actions are applied to form a new CS.
            Communication - Agents communicate their s values to coalition peers
            Payoff Distribution - assign payoff and check i.r satisfied for all players
                                  if not, then coalition gets 0 payoff as disagreement
        """
        # ------------------------------------------------------ #
        # movement phase
        next_state = self.movement_phase(\
            self.CS, self.locs, self.n, actions)

        # communication phase
        comm_vals, char_vals, sum_vals = self.communication_phase(\
            self.CS, self.sin_vals, self.n, self.cnf)

        # payoff distribution phase
        rewards = self.payoff_dist_phase(\
            comm_vals, char_vals, sum_vals, self.CS, self.locs, self.sin_vals, self.n)

        info = [self.CS]
        # compile next_state, reward, done, info and return
        return next_state, rewards, self.done, info
        # ------------------------------------------------------ #


    def reset(self, n, tasks, cnf=0.1):

        # Numerical Parameters:
        # ------------------------------------------------------ #
        self.n = n
        self.num_of_tasks = tasks
        self.done = False # dead variable
        self.cnf = cnf # communication noise factor of s value
        # ------------------------------------------------------ #

        # Data Arrays:
        # ------------------------------------------------------ #
        self.CS = [set() for i in range(1, self.num_of_tasks + 1)] # starting coalition structure
        self.locs = {} # track the location of each agent in the coalition structure
        self.sin_vals = {} # assigned singleton value to each agent
        # ------------------------------------------------------ #

        # Initialisation:
        # ------------------------------------------------------ #
        random.seed() # reset the seed
        for player in range(self.n): # choose a random task for each agent

            des_C = random.randint(0, self.num_of_tasks-1) # pick a random coalition...
            self.CS[des_C].add(f'{player +1 }') # ... and add the player to it...

            self.locs[f'Player {player + 1}'] = des_C # ... while recording the index...
            self.sin_vals[f'Player {player + 1}'] = random.random() # ...and its value
        # ------------------------------------------------------ #

        state = self.get_observations_from_CS(self.CS, self.locs, self.n)

        return state

    def render(self):
        pass
    # // -------------------- Gym Methods -------------------- #




