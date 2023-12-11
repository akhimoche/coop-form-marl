#!/usr/bin/env python
# coding: utf-8

# In[1]:

import random
import gym
import numpy as np

class CoopEnv(gym.Env):

    def __init__(self, n, num_of_tasks, cnf=0.1):


        # Numerical Parameters:
        # ------------------------------------------------------ #
        self.n = n
        self.num_of_tasks = num_of_tasks
        self.done = False
        self.cnf = cnf # communication noise factor of singleton value
        # ------------------------------------------------------ #


        # Data Arrays:
        # ------------------------------------------------------ #
        self.CS = [set() for i in range(1, self.num_of_tasks + 1)] # starting coalition structure
        self.player_locations = {} # track the location of each agent in the coalition structure
        self.singleton_vals = {}
        # ------------------------------------------------------ #


        # Initialisation:
        # ------------------------------------------------------ #
        for player in range(self.n): # choose a random task for each agent
            random.seed() # reset the seed
            chosen_task = random.randint(0, self.num_of_tasks-1) # pick a random coalition...
            self.CS[chosen_task].add(f'{player +1 }') # ... and add the player to it...

            self.player_locations[f'Player {player + 1}'] = chosen_task # ... while recording the index...
            self.singleton_vals[f'Player {player + 1}'] = 1 # ...and its value
        # ------------------------------------------------------ #

    # -------------------- Game Support Methods -------------------- # //
    def characteristic_function(self, coalition, seed):

        """ Return the characteristic value of a coalition
            Input: 'coalition' - Set of players present within a coalition of a coalition structure
            Output: 'value' - float representing value of input coalition

        """

        a = 3
        b = self.n/2
        c = self.n/5
        x = len(coalition)
        func_val = a*np.exp( -(x-b)**2 / (2*c**2)) # not necessarily superadditive...

        random.seed(seed) # original seeds with a shift for variety
        bias = random.uniform(func_val*0.8, func_val*1.2)

        if len(coalition) == 1: # ... but individual coalitions are always the singleton values
            bias = 1

        value = sum([self.singleton_vals[f'Player {player}'] for player in coalition]) * bias

        return value


    def get_observations_from_CS(self):

        """ Given a coalition structure, return ordered list of binary strings representing
            set observations for each agent (i.e what set the agent is in).

            Output: 'binary_list' - List of arrays containing binary string representation of
                                    each observed set as observed by each agent.

        """

        agent_observations = []

        for i in range(self.n):

            task = self.player_locations[f'Player {i+1}']
            current_coalition = self.CS[task] # get the coalition that player i is in

            binary_observation = np.zeros((4)) # prepare an observation array

            binary_observation[0] = len(current_coalition) # current coalition size
            binary_observation[1] = len(self.CS[(task-1+self.num_of_tasks)%self.num_of_tasks]) # left coalition size
            binary_observation[2] = len(self.CS[(task+1+self.num_of_tasks)%self.num_of_tasks]) # right coalition size
            binary_observation[3] = task # task number

            agent_observations.append(binary_observation)

        return agent_observations
    # // -------------------- Game Support Methods -------------------- #




    # -------------------- Game Phase Execution Methods -------------------- # //
    def movement_phase(self, actions):

        for player in range(self.n): # action will be index of task to join

            current_coalition = self.player_locations[f'Player {player + 1}'] # old coalition index

            # discern move (PBC)
            if actions[player] == 0: # hold position
                continue

            if actions[player] == 1: # move to left vertex
                new_coalition =  (current_coalition-1+self.num_of_tasks)%self.num_of_tasks

            if actions[player] == 2: # move to right vertex
                new_coalition = (current_coalition+1+self.num_of_tasks)%self.num_of_tasks

            # perform move
            self.CS[current_coalition].remove(f'{player + 1}') # remove from old coalition...
            self.CS[new_coalition].add(f'{player + 1}') # add player to new coalition...


            self.player_locations[f'Player {player + 1}'] = new_coalition # ... update locations


        next_state = self.get_observations_from_CS()
        return next_state

    def communication_phase(self):

        comm_vals = {}
        random.seed() # reset the seed for random communication noise (temporary)
        for player in range(self.n):

            singleton_val = self.singleton_vals[f'Player {player + 1}']
            noise = random.uniform(-self.cnf, self.cnf)
            comm_vals[f'Player {player + 1}'] = singleton_val * (1 + noise)

        # Get payoffs for coalitions from char func:
        char_vals = np.zeros((self.num_of_tasks))
        comm_tots = np.zeros((self.num_of_tasks))

        #print(f'svals {self.singleton_vals}')
        #print(f'commvals {comm_vals}')
        #print(self.CS)

        for task in range(self.num_of_tasks):

            # get coalition in coalition structure
            coalition = self.CS[task]

            if len(coalition) > 0:

                # characteristic value of the coalition
                seed = task*self.n + len(coalition) # unique seed for task and coalition
                char_val = self.characteristic_function(coalition, seed)
                char_vals[task] = char_val

                # sum of the communicated values of players
                tot = sum([comm_vals[f'Player {player}'] for player in coalition])
                comm_tots[task] = tot

            else:

                char_vals[task] = 0
                comm_tots[task] = 0

        #print(f'charvals {char_vals}')
        #print(f'commtots {comm_tots}')


        return comm_vals, char_vals, comm_tots

    def payoff_dist_phase(self, comm_vals, char_vals, comm_tots):

        rewards = np.zeros((self.n))
        for player in range(self.n): # determine how payoff be divided for players in coalitions

            # global values #
            location = self.player_locations[f'Player {player + 1}'] # player location (task index in CS)
            singleton_val = self.singleton_vals[f'Player {player + 1}'] # player singleton value

            # local values #
            comm_val = comm_vals[f'Player {player + 1}'] # communicated s. value
            coal_val = char_vals[location] # value of the coalition the player is in
            comm_sum = comm_tots[location] # sum of singleton values of coalition

            frac = (comm_val/comm_sum)

            if frac > 1:
                print("WARNING: Payoff fraction should not be greater than 1.")

            payoff = frac * coal_val
            rewards[player] = payoff

            # check stability with individual rationality
            if payoff < singleton_val:

                # all players in same coalition get 0 reward
                coalition = self.CS[location]
                indices = [int(s)-1 for s in coalition]
                rewards[indices] = 0

        return rewards
    # // -------------------- Game Phase Execution Methods -------------------- #




    # -------------------- Gym Methods -------------------- # //
    def step(self, actions):

        """ Three phases:
            Movement - Agents' chosen actions are applied to form a new CS.
            Communication - Agents communicate their singleton values to coalition peers
            Payoff Distribution - assign payoff and check i.r satisfied for all players
                                  if not, then coalition gets 0 payoff as disagreement
        """
        # Movement Phase
        next_state = self.movement_phase( actions)

        # Communication Phase
        comm_vals, char_vals, comm_tots = self.communication_phase()

        # Payoff Distribution Phase
        rewards = self.payoff_dist_phase(comm_vals, char_vals, comm_tots)

        info = []
        # compile next_state, reward, done, info and return
        return next_state, rewards, self.done, info
        # ------------------------------------------------------ #


    def reset(self, n, num_of_tasks , cnf=0.1):

        # Numerical Parameters:
        # ------------------------------------------------------ #
        self.n = n
        self.num_of_tasks = num_of_tasks
        self.done = False
        self.cnf = cnf # communication noise factor of singleton value
        # ------------------------------------------------------ #


        # Data Arrays:
        # ------------------------------------------------------ #
        self.CS = [set() for i in range(1, self.num_of_tasks + 1)] # starting coalition structure
        self.player_locations = {} # track the location of each agent in the coalition structure
        self.singleton_vals = {}
        # ------------------------------------------------------ #


        # Initialisation:
        # ------------------------------------------------------ #
        for player in range(self.n): # choose a random task for each agent
            random.seed() # reset the seed
            chosen_task = random.randint(0, self.num_of_tasks-1) # pick a random coalition...
            self.CS[chosen_task].add(f'{player +1 }') # ... and add the player to it...

            self.player_locations[f'Player {player + 1}'] = chosen_task # ... while recording the index...
            self.singleton_vals[f'Player {player + 1}'] = 1 # ...and its value
        # ------------------------------------------------------ #

        state = self.get_observations_from_CS()

        return state

    def render(self):
        pass
    # // -------------------- Gym Methods -------------------- #




