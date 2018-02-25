from collections import Counter
from random import randint

import numpy as np

from .q_learner import QLearner


class WhitelistLearner(QLearner):
    """A cautious agent that tries not to change the world too much in unknown ways."""
    unknown_cost = 250  # cost of each unknown change effected to the environment

    def __init__(self, examples, simulator):
        """Takes a series of state representations (training set) and a simulator."""
        self.whitelist = Counter()  # whitelist[char1, char2] := # of times char1 -> char2 observed during training
        for example in examples:
            for ind, state_b in enumerate(example[1:], start=1):  # for each of the t-1 transitory time slices
                self.whitelist.update(self.diff(example[ind-1], state_b))  # each observed transition

        super().__init__(simulator)  # do normal training

    def total_penalty(self, state_a, state_b):
        """Calculate the penalty incurred by the transition from state_a to state_b."""
        def penalty(square_a, square_b):
            """Using the whitelist counts, calculate penalty for square_a -> square_b."""
            return self.unknown_cost if square_a != square_b and (square_a, square_b) not in self.whitelist else 0

        return sum([penalty(tile_a, tile_b) for tile_a, tile_b in self.diff(state_a, state_b)])  # TODO numpy

    @staticmethod
    def diff(state_a, state_b):
        """Returns the string-wise transitions from state_a -> state_b."""
        diff = state_a != state_b
        return zip(state_a[diff], state_b[diff])

    def train(self, simulator):
        while self.num_samples.min() < self.convergence_bound:
            new_pos = randint(0, simulator.height - 1), randint(0, simulator.width - 1)

            # Go to new simulator state and take action
            simulator.reset()
            simulator.set_agent_pos(simulator.agent_pos, np.array(new_pos))
            reward = simulator.get_reward()  # reward in state[row][col]
            old_state = simulator.state.copy()

            action = self.e_greedy_action(new_pos)  # choose according to explore/exploit
            simulator.take_action(self.actions[action])
            self.num_samples[new_pos][action] += 1  # update sample count

            self.update_greedy(new_pos, action, reward - self.total_penalty(old_state, simulator.state), simulator)
