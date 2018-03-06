from random import choice as rchoice
from random import random, randint

import numpy as np


class QLearner:
    discount = 0.99  # how much it cares about future rewards
    epsilon = 0.25  # chance of choosing a random action
    convergence_bound = 200  # minimum number of tries for each (s, a) before terminating

    def __init__(self, simulator):
        """Trains using the simulator and e-greedy exploration to determine a greedy policy."""
        self.actions = simulator.get_actions()
        self.num_states = simulator.height * simulator.width
        self.Q = np.zeros((len(self.actions), simulator.height, simulator.width))
        self.greedy_a, self.greedy_v = np.zeros((simulator.height, simulator.width), int), \
                                       np.full((simulator.height, simulator.width), float('-inf'))

        self.num_samples = np.zeros((len(self.actions), simulator.height, simulator.width), int)
        for ind in range(len(self.actions)):
            self.num_samples[ind][tuple(simulator.goal_pos)] = self.convergence_bound  # don't bother with goal
        self.greedy_v[tuple(simulator.goal_pos)] = 0

        self.train(simulator)  # let's get to work!
        simulator.reset()  # clean up after ourselves

    def train(self, simulator):
        while self.num_samples.min() < self.convergence_bound:
            start_pos = randint(0, simulator.height - 1), randint(0, simulator.width - 1)
            if (start_pos == simulator.goal_pos).all():
                continue
            # Go to new simulator state and take action
            simulator.reset()
            simulator.agent_pos = np.array(start_pos)

            action = self.e_greedy_action(start_pos)  # choose according to explore/exploit
            reward = simulator.take_action(self.actions[action])
            self.num_samples[action][start_pos] += 1  # update sample count

            self.update_greedy(start_pos, action, reward, simulator)

    def e_greedy_action(self, pos):
        """Returns the e-greedy action index for the given position."""
        return self.get_other(self.greedy_a[pos], list(range(len(self.actions)))) if random() < self.epsilon \
            else self.greedy_a[pos]

    def update_greedy(self, start_pos, action, reward, simulator):
        """Perform TD update on observed reward; update greedy actions and values, if appropriate."""
        learning_rate = 1 / self.num_samples[action][start_pos]
        self.Q[action][start_pos] += learning_rate * (reward + self.discount * self.maxQ(simulator.agent_pos)
                                                      - self.Q[action][start_pos])

        if self.Q[action][start_pos] > self.greedy_v[start_pos]:
            self.greedy_a[start_pos], self.greedy_v[start_pos] = action, self.Q[action][start_pos]

    def maxQ(self, pos):
        largest = float('-inf')
        for ind in range(len(self.actions)):
            largest = max(largest, self.Q[ind][tuple(pos)])
        return largest

    def choose_action(self, state):
        return self.actions[self.greedy_a[tuple(state.agent_pos)]]

    def __str__(self):
        return "Q"

    @staticmethod
    def get_other(obj, objects):
        """Return another object in the Iterable besides the given one."""
        if len(objects) < 2: return obj  # make sure there *is* another object we can choose
        other = rchoice(objects)
        while other == obj:
            other = rchoice(objects)
        return other
