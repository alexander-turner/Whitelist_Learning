from collections import Counter
from random import randint

from .q_learner import QLearner


class WhitelistLearner(QLearner):
    """A cautious agent that tries not to change the world too much in unknown ways."""
    unknown_cost = 250  # cost of each unknown change effected to the environment

    def __init__(self, examples, simulator):
        """Takes a series of state representations (training set) and a simulator."""
        # whitelist[char1, char2] := number of times char1 -> char2 observed during training
        self.whitelist = Counter()
        for example in examples:
            for ind, state_b in enumerate(example[1:], start=1):  # for each of the t-1 transitory time slices
                self.whitelist.update(self.diff(example[ind-1], state_b))  # each observed transition

        super().__init__(simulator)  # do normal training

    def penalty(self, state_a, state_b):
        """Calculate the penalty incurred by the transition from state_a to state_b."""
        return sum([self.unknown_cost for difference in self.diff(state_a, state_b)
                    if difference not in self.whitelist])

    @staticmethod
    def diff(state_a, state_b):
        """Returns the string-wise transitions from state_a -> state_b."""
        return [(tile_a, tile_b) for row_a, row_b in zip(state_a, state_b) for tile_a, tile_b in zip(row_a, row_b)
                if tile_a != tile_b]

    def train(self, simulator):
        while self.num_samples.min() < self.convergence_bound:
            row, col = randint(0, simulator.height - 1), randint(0, simulator.width - 1)

            # Go to new simulator state and take action
            simulator.reset()
            simulator.set_agent_pos(simulator.agent_pos, [row, col])
            reward = simulator.get_reward()  # reward in state[row][col]
            old_state = [row.copy() for row in simulator.state]

            action = self.e_greedy_action(row, col)  # choose according to explore/exploit
            simulator.take_action(self.actions[action])
            self.num_samples[row][col][action] += 1  # update sample count

            self.update_greedy(row, col, action, reward - self.penalty(old_state, simulator.state), simulator)
