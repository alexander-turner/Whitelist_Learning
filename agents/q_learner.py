from random import random, randint

import numpy as np


class QLearner:
    discount = 0.9  # how much it cares about future rewards
    epsilon = 0.2  # chance of choosing a random action
    convergence_bound = 320  # minimum number of tries for each (s, a) before terminating

    def __init__(self, simulator):
        """Trains using the simulator and e-greedy exploration to determine a greedy policy."""
        self.actions = simulator.get_actions()
        self.num_states = simulator.height * simulator.width
        self.Q = np.zeros((simulator.height, simulator.width, len(self.actions)))
        self.greedy_a, self.greedy_v = np.zeros((simulator.height, simulator.width), int), \
                                       np.full((simulator.height, simulator.width), float('-inf'))

        self.num_samples = np.zeros((simulator.height, simulator.width, len(self.actions)), int)

        self.train(simulator)  # let's get to work!
        simulator.reset()  # clean up after ourselves

    def train(self, simulator):
        while self.num_samples.min() < self.convergence_bound:
            new_pos = randint(0, simulator.height - 1), randint(0, simulator.width - 1)

            # Go to new simulator state and take action
            simulator.reset()
            simulator.set_agent_pos(simulator.agent_pos, np.array(new_pos))
            reward = simulator.get_reward()

            action = self.e_greedy_action(new_pos)  # choose according to explore/exploit
            simulator.take_action(self.actions[action])
            self.num_samples[new_pos][action] += 1  # update sample count

            self.update_greedy(new_pos, action, reward, simulator)

    def e_greedy_action(self, new_pos):
        """Returns the e-greedy action index for the given position."""
        action = self.greedy_a[new_pos]
        if random() < self.epsilon:
            action = randint(0, len(self.actions) - 1)
            while action == self.greedy_a[new_pos]:  # make sure we don't choose the greedy action
                action = randint(0, len(self.actions) - 1)
        return action

    def update_greedy(self, new_pos, action, reward, simulator):
        """Perform TD update on observed reward; update greedy actions and values, if appropriate."""
        learning_rate = 1 / self.num_samples[new_pos][action]
        best_q = max(self.Q[simulator.agent_pos[0]][simulator.agent_pos[1]])
        self.Q[new_pos][action] += learning_rate * (reward + self.discount * best_q - self.Q[new_pos][action])

        if self.Q[new_pos][action] > self.greedy_v[new_pos]:
            self.greedy_a[new_pos], self.greedy_v[new_pos] = action, self.Q[new_pos][action]

    def choose_action(self, state):
        return self.actions[self.greedy_a[tuple(state.agent_pos)]]
