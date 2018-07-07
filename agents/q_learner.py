from random import choice as rchoice
from random import random, randint

import numpy as np


class QLearner:
    discount = 1  # how much it cares about future rewards
    epsilon = 0.6  # chance of choosing a random action in training

    def __init__(self, simulator):
        """Trains using the simulator and e-greedy exploration to determine a greedy policy."""
        self.actions = simulator.get_actions()
        self.num_states = simulator.height * simulator.width
        self.Q = np.zeros((len(self.actions), simulator.height, simulator.width))
        self.greedy_a, self.greedy_v = np.zeros((simulator.height, simulator.width), int), \
                                       np.full((simulator.height, simulator.width), float('-inf'))

        self.num_samples = np.zeros((len(self.actions), simulator.height, simulator.width), int)
        for ind in range(len(self.actions)):
            self.num_samples[ind][tuple(simulator.goal_pos)] = 1  # don't bother with goal
        self.greedy_v[tuple(simulator.goal_pos)] = 0

        self.train(simulator)  # let's get to work!
        simulator.reset()  # clean up after ourselves

    def train(self, simulator):
        simulator.reset()
        for episode in range(200):  # run the appropriate number of episodes
            simulator.run(self, learn=True)  # do online simulated learning
            simulator.reset()

    def observe_state(self, state):
        """Learner-specific method for getting observational data from simulator."""
        return state

    def total_penalty(self, state_a, state_b):
        return 0

    def choose_action(self, state):
        return self.greedy_a[tuple(state.agent_pos)]

    def behavior_action(self, pos):
        """Returns the e-greedy action index for the given position."""
        return self.get_other(self.greedy_a[pos], list(range(len(self.actions)))) if random() < self.epsilon \
            else self.greedy_a[pos]

    def update_greedy(self, old_pos, action, reward, new_pos):
        """Perform TD update on observed reward; update greedy actions and values, if appropriate."""
        learning_rate = .4
        self.Q[action][old_pos] += learning_rate * (reward + self.discount * self.maxQ(new_pos)
                                                    - self.Q[action][old_pos])

        if self.Q[action][old_pos] > self.greedy_v[old_pos]:
            self.greedy_a[old_pos], self.greedy_v[old_pos] = action, self.Q[action][old_pos]

    def maxQ(self, pos):
        largest = float('-inf')
        for ind in range(len(self.actions)):
            largest = max(largest, self.Q[ind][tuple(pos)])
        return largest

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
