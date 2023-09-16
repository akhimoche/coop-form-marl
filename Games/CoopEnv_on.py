#!/usr/bin/env python
# coding: utf-8

# In[1]:


import gym 
import numpy as np
import random
from copy import deepcopy


# In[11]:


class CoopEnv(gym.Env):
    
    def characteristic_function(self, coalition, singleton_vals):
        
        """ Return the sum of values of individual players in a coalition.
            Input: 'coalition' - Set of players present within a coalition of a coalition structure
                   'singleton_vals' - Dictionary containing singleton values for each player
            Output: 'value' - float representing value of input coalition
        
        """
        
        value = sum(singleton_vals[f'Player {player}'] for player in coalition) * len(coalition)
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
        generic_string = np.zeros((n))
        observations = [CS[locations[f'Player {i+1}']] for i in range(n)] # for each agent, get the coalition they occupy
        
        for subset in observations:
            
            a = np.copy(generic_string)
            for j in subset:
                
                a[int(j)-1] = 1
                
            binary_list.append(a)
            
        return binary_list
    
    
    def __init__(self, n, task_num):
        
        
        # Numerical Parameters:
        # -------------------- #
        self.n = n
        self.num_of_tasks = task_num
        self.done = False
        # -------------------- #
        
        
        # Data Arrays:
        # ---------- #
        self.CS = [set() for i in range(1, self.num_of_tasks + 1)] # starting coalition structure 
        self.coalition_dict = {} # track the location of each agent in the coalition structure 
        self.single_dict = {}
        # ---------- #
        
        
        # Initialisation:
        # -------------- #
        for player_number in range(self.n): # choose a random task for each agent
            
            desired_coalition = random.randint(0, self.num_of_tasks-1) # pick a random coalition...
            self.CS[desired_coalition].add(f'{player_number +1 }') # ... and add the player to it...
            
            self.coalition_dict[f'Player {player_number + 1}'] = desired_coalition # ... while recording the index...
            self.single_dict[f'Player {player_number + 1}'] = random.random() # ...and its value
        # -------------- #
        
        
    def step(self, actions): # expect 'actions' to be a dictionary list of actions for each player
        
        # Get new coalition structure:
        # --------------------------- #
        
        for agent in range(self.n): # action will be index of task to join
            
            new_coalition = actions[agent] # get new coalition index
            current_coalition = self.coalition_dict[f'Player {agent+ 1}'] # get old coalition index
            
           
            self.CS[current_coalition].remove(f'{agent +1 }') # remove from old coalition...
            self.CS[new_coalition].add(f'{agent +1 }') # add player to new coalition...


            self.coalition_dict[f'Player {agent + 1}'] = new_coalition # ... and update coalition map

            
        next_state = self.get_observations_from_CS(self.CS, self.coalition_dict, self.n)
        # --------------------------- #
        
        
        # Observe payoffs for coalition structure:
        # --------------------------------------- #
        char_vals = []
        for subset in self.CS:
            
            if len(subset) > 0:
                
                value = self.characteristic_function(subset, self.single_dict)
                char_vals.append(value)
                
            else:
                
                char_vals.append(0)
        
        rewards = []
        for player in range(self.n):
            
            location = self.coalition_dict[f'Player {player + 1}']
            coalition_val = char_vals[location]
            reward = coalition_val/len(self.CS[location]) # assume reward is equal dist of coalition val
            rewards.append(reward)
        # --------------------------------------- #
        
        
        # Check termination and return next step info:
        # ------------------------------------------- #    
        info = [self.CS]
            
        # compile next_state, reward, done, info and return
        return next_state, rewards, self.done, info
        # ------------------------------------------- #
        
        
    def reset(self, n, task_num):
        
        
        # Numerical Parameters:
        # -------------------- #
        self.n = n
        self.num_of_tasks = task_num
        self.done = False
        # -------------------- #
        
        
        # Data Arrays:
        # ---------- #
        self.CS = [set() for i in range(1, self.num_of_tasks + 1)] # starting coalition structure 
        self.coalition_dict = {} # track the location of each agent in the coalition structure 
        self.single_dict = {}
        # ---------- #
        
        
        # Initialisation:
        # -------------- #
        for player_number in range(self.n): # choose a random task for each agent
            
            desired_coalition = random.randint(0, self.num_of_tasks-1) # pick a random coalition...
            self.CS[desired_coalition].add(f'{player_number +1 }') # ... and add the player to it...
            
            self.coalition_dict[f'Player {player_number + 1}'] = desired_coalition # ... while recording the index...
            self.single_dict[f'Player {player_number + 1}'] = random.random() # ...and its value
        # -------------- #
        
        state = self.get_observations_from_CS(self.CS, self.coalition_dict, self.n)
        
        return state
    
    def render(self):
        pass





