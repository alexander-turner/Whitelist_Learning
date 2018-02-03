from .q_learner import QLearner
from collections import Counter
from copy import deepcopy
from random import random, randint


class WhitelistLearner(QLearner):
    """A cautious agent that tries not to change the world too much (unlike its creator)."""
    unknown_cost = 50  # cost to each unknown change effected to the environment

    def __init__(self, examples, simulator):
        """Takes a flattened list of observed tile transitions during training episodes and a simulator."""
        # whitelist[char1, char2] := number of times char1 -> char2 observed during training
        self.whitelist = Counter()
        for example in examples:
            for char1, char2 in example:  # each observed transition
                self.whitelist[char1, char2] += 1

        super().__init__(simulator)  # do normal training

    def grade_transition(self, state_a, state_b):
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
        while self.num_samples.min() < self.convergence_bound:
            row, col = randint(0, simulator.height - 1), randint(0, simulator.width - 1)

            # Choose according to explore/exploit
            if random() < self.epsilon:
                action = randint(0, len(self.actions) - 1)
                while action == self.greedy_a[row][col]:  # make sure we don't choose greedy action
                    action = randint(0, len(self.actions) - 1)
            else:
                action = self.greedy_a[row][col]

            # Update sample count and learning rate
            self.num_samples[row][col][action] += 1
            learning_rate = 1 / self.num_samples[row][col][action]

            # Go to new simulator state and take action
            simulator.reset()  # TODO how do we track distance to goal?
            simulator.set_agent_pos(simulator.agent_pos, (row, col))
            simulator.agent_pos = [row, col]

            reward = simulator.get_reward()
            old_state = deepcopy(simulator.state)
            simulator.take_action(self.actions[action])
            new_state = simulator.agent_pos
            penalty = self.grade_transition(old_state, simulator.state)

            # Perform TD update
            self.Q[row][col][action] += learning_rate * (
                reward + self.discount * max(self.Q[new_state[0]][new_state[1]])
                - self.Q[row][col][action] - penalty)

            # See if this is better than state's current greedy action
            if self.Q[row][col][action] > self.greedy_v[row][col]:
                self.greedy_a[row][col], self.greedy_v[row][col] = action, self.Q[row][col][action]
