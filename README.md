# Multi-Agent Reinforcement Learning for Coalitional Formation Games

This repository contains all files related to simulating the coalitional formation games and applying MARL algorithms to solve for the stable coalition structure. This is the base model:

- Characteristic function is strictly superadditive; the sum of singleton values of constituent players is multiplied by length of the coalition. Said singleton values are pulled randomly between 0 and 1.
- Payoffs are distributed according to each player's (in a coalition) singleton value if individual rationality is satisfied for each player.
- No strategic decision making for payoff distribution, the calculation is made centrally without interaction from the players.  
