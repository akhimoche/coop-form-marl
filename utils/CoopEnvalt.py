#!/usr/bin/env python
# coding: utf-8

# In[1]:

import random
import gym
import numpy as np

class CoopEnv(gym.Env):

    # -------------------- Game Support Methods -------------------- # //
    def characteristic_function(self, coalition, singleton_vals, seed):

        """ Return the characteristic value of a coalition
            Input: 'coalition' - Set of players present within a coalition of a coalition structure
                   'singleton_vals' - Dictionary containing singleton values for each player
            Output: 'value' - float representing value of input coalition

        """

        random.seed(seed) # original seeds with a shift for variety
        bias = random.uniform(0,len(coalition)) # not necessarily superadditive...

        if len(coalition) == 1: # ... but individual coalitions are always the singleton values
            bias = 1

        value = sum(singleton_vals[f'Player {player}'] for player in coalition) * bias

        return value


    def get_observations_from_CS(self, CS, locations, n, tasks):

        """ Given a coalition structure, return ordered list of binary strings representing
            set observations for each agent (i.e what set the agent is in).
            Input: 'CS' - List of sets representing current coalition structure of the game.
                   'locations' - Dictionary returning index representing occupied coalition index.
                   'n' - Number of players.
            Output: 'binary_list' - List of arrays containing binary string representation of
                                    each observed set as observed by each agent.

        """

        states_list = []

        for i in range(n):

            loc = locations[f'Player {i+1}']
            subset = CS[loc]
            obs_vector = np.zeros((n+tasks))

            player_indices = [int(a)-1 for a in list(subset)] # indices are 1 less than the player tag
            coalition_index = loc + n # the one hot vector is appended so +n as loc is already an index
            obs_vector[player_indices] = 1
            obs_vector[coalition_index] = 1
            states_list.append(obs_vector)

        return states_list
    # // -------------------- Game Support Methods -------------------- #



    # -------------------- Game Phase Execution Methods -------------------- # //
    def movement_phase(self, CS, locations, n, tasks, actions):

        for player in range(n): # action will be index of task to join

            new_coalition = int(actions[player]) # get new coalition index
            current_coalition = locations[f'Player {player + 1}'] # old coalition index

            CS[current_coalition].remove(f'{player + 1}') # remove from old coalition...
            CS[new_coalition].add(f'{player + 1}') # add player to new coalition...


            locations[f'Player {player + 1}'] = new_coalition # ... update locations


        next_state = self.get_observations_from_CS(CS, locations, n, tasks)
        return next_state

    def communication_phase(self, CS, singleton_vals, n, actions):

        comm_vals = {}
        for player in range(n):

            singleton_val = singleton_vals[f'Player {player + 1}']
            comm_vals[f'Player {player + 1}'] = singleton_val * (1 + actions[player])

        # Get payoffs for coalitions from char func:
        char_vals = []
        comm_tots = []
        task = 0
        for coalition in CS:

            # get value for each coalition in coalition structure
            if len(coalition) > 0:

                # characteristic value of the coalition
                seed = task*n + len(coalition) # unique seed for task and coalition
                char_value = self.characteristic_function(coalition, singleton_vals, seed)
                char_vals.append(char_value)

                # sum of the communicated values of players
                tot = sum(comm_vals[f'Player {player}'] for player in coalition)
                comm_tots.append(tot)

            else:

                char_vals.append(0)
                comm_tots.append(0)

            task += 1

        return comm_vals, char_vals, comm_tots

    def payoff_dist_phase(self, comm_vals, char_vals, comm_tots, CS, locations, singleton_vals, n):

        rewards = np.zeros((n))
        for player in range(n): # determine how payoff be divided for players in coalitions

            # global values #
            location = locations[f'Player {player + 1}'] # player location (task index in CS)
            singleton_val = singleton_vals[f'Player {player + 1}'] # player singleton value

            # local values #
            comm_val = comm_vals[f'Player {player + 1}'] # communicated s. value
            coal_val = char_vals[location] # value of the coalition the player is in
            comm_sum = comm_tots[location] # sum of singleton values of coalition

            payoff = (comm_val/comm_sum) * coal_val
            rewards[player] = payoff

            # check stability with individual rationality
            if payoff < singleton_val:

                # if an player doesn't like coalition
                # all players in same coalition get 0 reward
                coalition = CS[location]
                indices = [int(s)-1 for s in coalition]
                rewards[indices] = 0

        return rewards
    # // -------------------- Game Phase Execution Methods -------------------- #



    # -------------------- Gym Methods -------------------- # //
    def __init__(self, n, tasks):


        # Numerical Parameters:
        # ------------------------------------------------------ #
        self.n = n
        self.tasks = tasks
        self.done = False
        # ------------------------------------------------------ #


        # Data Arrays:
        # ------------------------------------------------------ #
        self.CS = [set() for i in range(1, self.tasks + 1)] # starting coalition structure
        self.player_locations = {} # track the location of each agent in the coalition structure
        self.singleton_vals = {}
        # ------------------------------------------------------ #


        # Initialisation:
        # ------------------------------------------------------ #
        for player in range(self.n): # choose a random task for each agent
            random.seed(player)# unique seed for each player
            chosen_task = random.randint(0, self.tasks-1) # pick a random coalition...
            self.CS[chosen_task].add(f'{player +1 }') # ... and add the player to it...

            self.player_locations[f'Player {player + 1}'] = chosen_task # ... record index...
            self.singleton_vals[f'Player {player + 1}'] = random.random() # ...and its value
        # ------------------------------------------------------ #


    def step(self, actions):

        """ Three phases:
            Movement - Agents' chosen actions are applied to form a new CS.
            Communication - Agents communicate their singleton values to coalition peers
            Payoff Distribution - assign payoff and check i.r satisfied for all players
                                  if not, then coalition gets 0 payoff as disagreement
        """
        # Movement Phase - first column is agent movement actions
        next_state = self.movement_phase(self.CS, self.player_locations, self.n, self.tasks, actions[:,0])

        # Communication Phase
        comm_vals, char_vals, comm_tots = self.communication_phase(self.CS, self.singleton_vals, self.n, actions[:,1])

        # Payoff Distribution Phase
        rewards = self.payoff_dist_phase(comm_vals, char_vals, comm_tots, self.CS, self.player_locations, self.singleton_vals, self.n)

        info = [self.CS]
        # compile next_state, reward, done, info and return
        return next_state, rewards, self.done, info
        # ------------------------------------------------------ #


    def reset(self, n, tasks):

        # Numerical Parameters:
        # ------------------------------------------------------ #
        self.n = n
        self.tasks = tasks
        self.done = False
        # ------------------------------------------------------ #


        # Data Arrays:
        # ------------------------------------------------------ #
        self.CS = [set() for i in range(1, self.tasks + 1)] # starting coalition structure
        self.player_locations = {} # track the location of each agent in the coalition structure
        self.singleton_vals = {}
        # ------------------------------------------------------ #


        # Initialisation:
        # ------------------------------------------------------ #
        for player in range(self.n): # choose a random task for each agent
            random.seed(player)# unique seed for each player
            chosen_task = random.randint(0, self.tasks-1) # pick a random coalition...
            self.CS[chosen_task].add(f'{player +1 }') # ... and add the player to it...

            self.player_locations[f'Player {player + 1}'] = chosen_task # ... while recording the index...
            self.singleton_vals[f'Player {player + 1}'] = random.random() # ...and its value
        # ------------------------------------------------------ #

        state = self.get_observations_from_CS(self.CS, self.player_locations, self.n, self.tasks)

        return state

    def render(self):
        pass
    # // -------------------- Gym Methods -------------------- #



