from random import random, randint

import numpy as np


class QLearner:
    discount = 0.99  # how much it cares about future rewards
    epsilon = 0.25  # chance of choosing a random action
    convergence_bound = 100  # minimum number of tries for each (s, a) before terminating

    def __init__(self, simulator):
        """Trains using the simulator and e-greedy exploration to determine a greedy policy."""
        self.actions = simulator.get_actions()
        self.num_states = simulator.height * simulator.width
        self.Q = np.zeros((simulator.height, simulator.width, len(self.actions)))
        self.greedy_a, self.greedy_v = np.zeros((simulator.height, simulator.width), int), \
                                       np.full((simulator.height, simulator.width), float('-inf'))

        self.num_samples = np.zeros((simulator.height, simulator.width, len(self.actions)), int)
        self.num_samples[tuple(simulator.goal_pos)].fill(self.convergence_bound)  # don't bother with goal square
        self.greedy_v[tuple(simulator.goal_pos)].fill(0)

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
            self.num_samples[start_pos][action] += 1  # update sample count

            self.update_greedy(start_pos, action, reward, simulator)

    def e_greedy_action(self, pos):
        """Returns the e-greedy action index for the given position."""
        action = self.greedy_a[pos]
        if random() < self.epsilon:
            action = randint(0, len(self.actions) - 1)
            while action == self.greedy_a[pos]:  # make sure we don't choose the greedy action
                action = randint(0, len(self.actions) - 1)
        return action

    def update_greedy(self, start_pos, action, reward, simulator):
        """Perform TD update on observed reward; update greedy actions and values, if appropriate."""
        learning_rate = 1 / self.num_samples[start_pos][action]
        self.Q[start_pos][action] += learning_rate * (reward + self.discount * max(self.Q[tuple(simulator.agent_pos)])
                                                      - self.Q[start_pos][action])

        if self.Q[start_pos][action] > self.greedy_v[start_pos]:
            self.greedy_a[start_pos], self.greedy_v[start_pos] = action, self.Q[start_pos][action]

    def choose_action(self, state):
        return self.actions[self.greedy_a[tuple(state.agent_pos)]]
