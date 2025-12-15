## Multi-Agent Reinforcement Learning for Cooperative Games

Cooperative games, or coalitional formation games, involve finding the set of coalition structures (that is, the division of N players into disjoint subsets) wherein no player could achieve better payoff in another coalition. This requirement for core stability is strong and becomes exponentially intractable as N increases - therefore finding a decentralised method to search for the optimal coalition structure(s) is needed.

This project proposes using independent multi-agent reinforcement learning as such a method. This is the main repository containing all present research materials. The branches contain different cooperative games denoted as below:

- 'base' - The core game without any communication between the players - all payoff generated from a particular coalition structure is divided according to the weight assigned to each agent
- 'djcomm' - An ability of each agent to lie about their payoff is added so that the payoff can be divided unfairly. All branches except base have this present.
- 'TS' - Thomspon Sampling is used to select the 'lying band' for each agent. It is done discretely rather than continuously to avoid numerical errors in the Thompson sampling.
- 'TP' - Task preferences, agents prefer to be in a particular coalition compared to others.
- 'dummy' - Contains zealot agents along with MARL agents in simulation.
- 'chaotic' - Contains chaotic agents along with MARL agents in simulation.
- 'mixed' - Contains all types of agents in simulation.
- 'simplegreedy' - Contains heuristic agents to benchmark MARL agent self-organisation capabilities.

To understand the context of this project, please see the following article: (https://akhimocherla.substack.com/p/coalition-structure-generation-with)
