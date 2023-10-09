#!/usr/bin/env python
# coding: utf-8

# In[1]:

import random
import gym
import numpy as np

class CoopEnv(gym.Env):

    def characteristic_function(self, coalition, singleton_vals, task, n):

        """ Return the characteristic value of a coalition
            Input: 'coalition' - Set of players present within a coalition of a coalition structure
                   'singleton_vals' - Dictionary containing singleton values for each player
            Output: 'value' - float representing value of input coalition

        """

        a = task*n + len(coalition)
        shift = 1
        length = len(coalition)
        random.seed(a+shift) # original seeds with a shift for variety
        bias = random.uniform(0,length) # not necessarily superadditive...

        if length == 1: # ... but individual coalitions are always the singleton values
            bias = 1

        value = sum(singleton_vals[f'Player {player}'] for player in coalition) * bias

        return value

    def get_sum(self, coalition, singleton_vals):

        """ Return the sum of values of individual players in a coalition.
            Input: 'coalition' - Set of players present within a coalition of a coalition structure
                   'singleton_vals' - Dictionary containing singleton values for each player
            Output: 'value' - float representing value of input coalition

        """

        value = sum(singleton_vals[f'Player {player}'] for player in coalition)
        return value


    def get_observations_from_CS(self, CS, locations, n):

        """ Given a coalition structure, return ordered list of binary strings representing
            set observations for each agent (i.e what set the agent is in).
            Input: 'CS' - List of sets representing current coalition structure of the game.
                   'locations' - Dictionary returning index representing occupied coalition index.
                   'n' - Number of players.
            Output: 'binary_list' - List of arrays containing binary string representation of
                                    each observed set as observed by each agent.

        """

        binary_list = []

        for i in range(n):

            subset = CS[locations[f'Player {i+1}']]
            binary_observation = np.zeros((n))

            indices = [int(a)-1 for a in list(subset)] # indices are 1 less than the player tag
            binary_observation[indices] = 1
            binary_list.append(binary_observation)

        return binary_list


    def __init__(self, n, tasks, cnf=0.1):


        # Numerical Parameters:
        # ------------------------------------------------------ #
        self.n = n
        self.num_of_tasks = tasks
        self.done = False
        self.cnf = cnf # communication noise factor of singleton value
        # ------------------------------------------------------ #


        # Data Arrays:
        # ------------------------------------------------------ #
        self.CS = [set() for i in range(1, self.num_of_tasks + 1)] # starting coalition structure
        self.coalition_dict = {} # track the location of each agent in the coalition structure
        self.single_dict = {}
        # ------------------------------------------------------ #


        # Initialisation:
        # ------------------------------------------------------ #
        random.seed() # reset the seed
        for player in range(self.n): # choose a random task for each agent

            desired_coalition = random.randint(0, self.num_of_tasks-1) # pick a random coalition...
            self.CS[desired_coalition].add(f'{player +1 }') # ... and add the player to it...

            self.coalition_dict[f'Player {player + 1}'] = desired_coalition # ... record index...
            self.single_dict[f'Player {player + 1}'] = random.random() # ...and its value
        # ------------------------------------------------------ #


    def step(self, actions):

        """ Three phases:
            Movement - Agents' chosen actions are applied to form a new CS.
            Communication - Agents communicate their singleton values to coalition peers
            Payoff Distribution - assign payoff and check i.r satisfied for all players
                                  if not, then coalition gets 0 payoff as disagreement
        """

        # MOVEMENT PHASE:
        # ------------------------------------------------------ #
        for player in range(self.n): # action will be index of task to join

            new_coalition = actions[player] # get new coalition index
            current_coalition = self.coalition_dict[f'Player {player + 1}'] # old coalition index

            self.CS[current_coalition].remove(f'{player + 1}') # remove from old coalition...
            self.CS[new_coalition].add(f'{player + 1}') # add player to new coalition...


            self.coalition_dict[f'Player {player + 1}'] = new_coalition # ... update locations


        next_state = self.get_observations_from_CS(self.CS, self.coalition_dict, self.n)
        # ------------------------------------------------------ #

        # COMMUNICATION PHASE:
        # ------------------------------------------------------ #
        communicated_vals = {}
        random.seed() # reset the seed for random communication noise (temporary)
        for player in range(self.n):

            singleton_val = self.single_dict[f'Player {player + 1}']
            noise = random.uniform(-self.cnf, self.cnf)
            communicated_vals[f'Player {player + 1}'] = singleton_val * (1 + noise)

        # Get payoffs for coalitions from char func:
        coal_char_vals = []
        coal_sum_vals = []
        task = 0
        for coalition in self.CS:

            # get value for each coalition in coalition structure
            if len(coalition) > 0:

                char_value = self.characteristic_function(coalition, self.single_dict, task, self.n)
                coal_char_vals.append(char_value)

                sum_value = self.get_sum(coalition, communicated_vals)
                coal_sum_vals.append(sum_value)

            else:

                coal_char_vals.append(0)
                coal_sum_vals.append(0)

            task += 1
        # ------------------------------------------------------ #

        # PAYOFF DISTRIBUTION PHASE:
        # ------------------------------------------------------ #
        rewards = np.zeros((self.n))
        for player in range(self.n): # determine how payoff be divided for players in coalitions

            # global values #
            location = self.coalition_dict[f'Player {player + 1}'] # player location (task index in CS)
            singleton_val = self.single_dict[f'Player {player + 1}'] # player singleton value

            # local values #
            comm_val = communicated_vals[f'Player {player + 1}'] # communicated s. value
            coal_val = coal_char_vals[location] # value of the coalition the player is in
            coal_comm_sum = coal_sum_vals[location] # sum of singleton values of coalition

            payoff = (comm_val/coal_comm_sum) * coal_val
            rewards[player] = payoff

            # check stability with individual rationality
            if payoff < singleton_val:

                # if an agent doesn't like coalition
                # all agents get 0 reward
                coalition = self.CS[location]
                indices = [int(s)-1 for s in coalition]
                rewards[indices] = 0
        # ------------------------------------------------------ #


        # ------------------------------------------------------ #
        # Check termination and return next step info:
        # ------------------------------------------------------ #
        info = [self.CS]

        # compile next_state, reward, done, info and return
        return next_state, rewards, self.done, info
        # ------------------------------------------------------ #


    def reset(self, n, tasks, cnf=0.1):

        # Numerical Parameters:
        # ------------------------------------------------------ #
        self.n = n
        self.num_of_tasks = tasks
        self.done = False
        self.cnf = cnf # communication noise factor of singleton value
        # ------------------------------------------------------ #


        # Data Arrays:
        # ------------------------------------------------------ #
        self.CS = [set() for i in range(1, self.num_of_tasks + 1)] # starting coalition structure
        self.coalition_dict = {} # track the location of each agent in the coalition structure
        self.single_dict = {}
        # ------------------------------------------------------ #


        # Initialisation:
        # ------------------------------------------------------ #
        random.seed() # reset the seed
        for player in range(self.n): # choose a random task for each agent

            desired_coalition = random.randint(0, self.num_of_tasks-1) # pick a random coalition...
            self.CS[desired_coalition].add(f'{player +1 }') # ... and add the player to it...

            self.coalition_dict[f'Player {player + 1}'] = desired_coalition # ... while recording the index...
            self.single_dict[f'Player {player + 1}'] = random.random() # ...and its value
        # ------------------------------------------------------ #

        state = self.get_observations_from_CS(self.CS, self.coalition_dict, self.n)

        return state

    def render(self):
        pass




