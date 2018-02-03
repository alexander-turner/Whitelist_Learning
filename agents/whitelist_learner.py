from collections import Counter
from copy import deepcopy
from random import randint

from .q_learner import QLearner


class WhitelistLearner(QLearner):
    """A cautious agent that tries not to change the world too much in unknown ways."""
    unknown_cost = 100  # cost of each unknown change effected to the environment TODO double-counting transitions?

    def __init__(self, examples, simulator):
        """Takes a series of state representations (training set) and a simulator."""
        # whitelist[char1, char2] := number of times char1 -> char2 observed during training
        self.whitelist = Counter()
        for example in examples:
            for ind, state_b in enumerate(example[1:], start=1):  # for each of the t-1 transitory time slices
                state_a = example[ind-1]
                for char1, char2 in self.diff(state_a, state_b):  # each observed transition
                    self.whitelist[char1, char2] += 1

        super().__init__(simulator)  # do normal training

    def penalty(self, state_a, state_b):
        """Calculate the penalty incurred by the transition from state_a to state_b."""
        penalty = 0
        for difference in self.diff(state_a, state_b):
            if difference not in self.whitelist:  # TODO make more sophisticated
                penalty += self.unknown_cost
        return penalty

    @staticmethod
    def diff(state_a, state_b):
        """Returns the string-wise transitions from state_a -> state_b."""
        differences = []
        for row_a, row_b in zip(state_a, state_b):
            for tile_a, tile_b in zip(row_a, row_b):
                if tile_a != tile_b:
                    differences.append((tile_a, tile_b))
        return differences

    def train(self, simulator):
        def rand_pos():
            """Return a random position on the game board."""
            return randint(0, simulator.height - 1), randint(0, simulator.width - 1)

        while self.num_samples.min() < self.convergence_bound:
            row, col = rand_pos()

            # Choose according to explore/exploit
            action = self.e_greedy_action(row, col)

            # Update sample count and learning rate
            self.num_samples[row][col][action] += 1
            learning_rate = 1 / self.num_samples[row][col][action]

            # Go to new simulator state and take action
            simulator.reset()
            simulator.set_agent_pos(simulator.agent_pos, [row, col])
            reward = simulator.get_reward()  # reward in state[row][col]

            old_state = deepcopy(simulator.state)
            simulator.take_action(self.actions[action])

            penalty = self.penalty(old_state, simulator.state)

            # Perform TD update
            self.Q[row][col][action] += learning_rate * (reward - penalty +
                                                         self.discount * max(
                                                             self.Q[simulator.agent_pos[0]][simulator.agent_pos[1]])
                                                         - self.Q[row][col][action])

            # See if this is better than state's current greedy action
            if self.Q[row][col][action] > self.greedy_v[row][col]:
                self.greedy_a[row][col], self.greedy_v[row][col] = action, self.Q[row][col][action]
