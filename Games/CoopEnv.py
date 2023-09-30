#!/usr/bin/env python
# coding: utf-8

# In[1]:


import gym 
import numpy as np
import random
from copy import deepcopy

class CoopEnv(gym.Env):
    
    def characteristic_function(self, coalition, singleton_vals):
        
        """ Return the characteristic value of a coalition
            Input: 'coalition' - Set of players present within a coalition of a coalition structure
                   'singleton_vals' - Dictionary containing singleton values for each player
            Output: 'value' - float representing value of input coalition
        
        """
        
        value = sum(singleton_vals[f'Player {player}'] for player in coalition) * len(coalition)
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
        
        """ Given a coalition structure, return ordered list of binary strings representing set observations
            for each agent (i.e what set the agent is in)
            Input: 'CS' - List of sets representing current coalition structure of the game.
                   'locations' - Dictionary returning index representing occupied coalition index.
                   'n' - Number of players.
            Output: 'binary_list' - List of arrays containing binary string representation of each observed set
                                     as observed by each agent.
        
        """
        
        binary_list = []
        
        for i in range(n):

            subset = CS[locations[f'Player {i+1}']]
            binary_observation = np.zeros((n))

            indices = [int(a)-1 for a in list(subset)] # indices are 1 less than the player tag
            binary_observation[indices] = 1
            binary_list.append(binary_observation)

        return binary_list
    
    
    def __init__(self, n, task_num, cnf=0.1):
        
        
        # Numerical Parameters:
        # -------------------- #
        self.n = n
        self.num_of_tasks = task_num
        self.done = False
        self.cnf = cnf # communication noise factor of singleton value 
        # -------------------- #
        
        
        # Data Arrays:
        # ---------- #
        self.CS = [set() for i in range(1, self.num_of_tasks + 1)] # starting coalition structure 
        self.coalition_dict = {} # track the location of each agent in the coalition structure 
        self.single_dict = {}
        self.stability_dict = {}
        # ---------- #
        
        
        # Initialisation:
        # -------------- #
        for player in range(self.n): # choose a random task for each agent
            
            desired_coalition = random.randint(0, self.num_of_tasks-1) # pick a random coalition...
            self.CS[desired_coalition].add(f'{player +1 }') # ... and add the player to it...
            
            self.coalition_dict[f'Player {player + 1}'] = desired_coalition # ... while recording the index...
            self.single_dict[f'Player {player + 1}'] = random.random() # ...and its value ...
            self.stability_dict[f'Player {player + 1}'] = False # ... while setting all stability to false
        # -------------- #
        
        
    def step(self, actions): # expect 'actions' to be a dictionary list of actions for each player
        
        # Get new coalition structure:
        # --------------------------- #
        for player in range(self.n): # action will be index of task to join
            
            new_coalition = actions[player] # get new coalition index
            current_coalition = self.coalition_dict[f'Player {player + 1}'] # get old coalition index
            
           
            self.CS[current_coalition].remove(f'{player + 1}') # remove from old coalition...
            self.CS[new_coalition].add(f'{player + 1}') # add player to new coalition...


            self.coalition_dict[f'Player {player + 1}'] = new_coalition # ... and update coalition map

            
        next_state = self.get_observations_from_CS(self.CS, self.coalition_dict, self.n)
        # --------------------------- #
        
        # Communicate singleton values between coalition players: 
        # ------------------------------------------------------ #
        communicated_vals = {}
        for player in range(self.n):
            single_val = self.single_dict[f'Player {player + 1}']
            communicated_vals[f'Player {player + 1}'] = single_val * (1 + random.uniform(-self.cnf, self.cnf)) # value of singleton (100%) +- cnf (as percentage) 
        # ------------------------------------------------------ #
        
        # Get payoffs for coalitions from char func:
        # --------------------------------------- #
        coal_char_vals = []
        real_comm_sum = []
        for coalition in self.CS: # get value for each coalition in coalition structure
            
            if len(coalition) > 0:
                
                full_value = self.characteristic_function(coalition, self.single_dict)
                coal_char_vals.append(full_value)
                
                sum_value = self.get_sum(coalition, communicated_vals)
                real_comm_sum.append(sum_value)
                
            else:
                
                coal_char_vals.append(0)
                real_comm_sum.append(0)
        # --------------------------------------- #
                
        rewards = []
        for player in range(self.n): # determine how this will be divided for players within the coalitions
            
            location = self.coalition_dict[f'Player {player + 1}'] # get location of player
            real_val = self.single_dict[f'Player {player + 1}'] # individual player real singleton value
            comm_val = communicated_vals[f'Player {player + 1}'] # individual player communicated singleton value
            
            coal_val = coal_char_vals[location] # value of the coalition the player is in
            coal_comm_sum = real_comm_sum[location] # sum of singleton values of coalition
            
            award = (comm_val/coal_comm_sum) * coal_val
            
            # check stability with individual rationality
            if award >= real_val:
            
                reward = award  
                self.stability_dict[f'Player {player + 1}'] = True
                
            else:
                
                reward = 0
                
            rewards.append(reward)
        # --------------------------------------- #
        # Check termination and return next step info:
        # ------------------------------------------- #    
        info = [self.CS]
            
        # compile next_state, reward, done, info and return
        return next_state, rewards, self.done, info
        # ------------------------------------------- #
        
        
    def reset(self, n, task_num, cnf=0.1):
        
        # Numerical Parameters:
        # -------------------- #
        self.n = n
        self.num_of_tasks = task_num
        self.done = False
        self.cnf = cnf # communication noise factor of singleton value 
        # -------------------- #
        
        
        # Data Arrays:
        # ---------- #
        self.CS = [set() for i in range(1, self.num_of_tasks + 1)] # starting coalition structure 
        self.coalition_dict = {} # track the location of each agent in the coalition structure 
        self.single_dict = {}
        self.stability_dict = {}
        # ---------- #
        
        
        # Initialisation:
        # -------------- #
        for player in range(self.n): # choose a random task for each agent
            
            desired_coalition = random.randint(0, self.num_of_tasks-1) # pick a random coalition...
            self.CS[desired_coalition].add(f'{player +1 }') # ... and add the player to it...
            
            self.coalition_dict[f'Player {player + 1}'] = desired_coalition # ... while recording the index...
            self.single_dict[f'Player {player + 1}'] = random.random() # ...and its value ...
            self.stability_dict[f'Player {player + 1}'] = False # ... while setting all stability to false
        # -------------- #
        
        state = self.get_observations_from_CS(self.CS, self.coalition_dict, self.n)

        return state
    
    def render(self):
        pass




