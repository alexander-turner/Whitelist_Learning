import numpy as np
from random import random, randint


class QLearner:
    discount = 0.9  # how much it cares about future rewards
    epsilon = 0.1  # chance of choosing a random-action
    convergence_bound = 100  # minimum number of tries for each (s, a) before terminating

    def __init__(self, simulator):
        """Trains using the simulator and e-greedy exploration to determine a greedy policy."""
        self.actions = simulator.get_actions()
        self.num_states = simulator.height * simulator.width
        self.Q = np.zeros((simulator.height, simulator.width, len(self.actions)))  # TODO check states
        self.greedy_a, self.greedy_v = np.ones((simulator.height, simulator.height), int), \
                                       np.zeros((simulator.height, simulator.height))  # greedy recordkeeping
        self.num_samples = np.zeros((simulator.height, simulator.width, len(self.actions)), int)

        self.train(simulator)  # let's get to work!
        simulator.reset()  # clean up after ourselves

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

            # Take action
            simulator.reset()  # TODO less?
            simulator.agent_position = [row, col]
            reward = simulator.get_reward()
            new_reward = simulator.take_action(self.actions[action])
            new_state = simulator.agent_position

            # Perform TD update
            self.Q[row][col][action] += learning_rate * (reward + self.discount * max(self.Q[new_state[0]][new_state[1]])
                                                         - self.Q[row][col][action])

            # See if this is better than state's current greedy action
            if self.Q[row][col][action] > self.greedy_v[row][col]:
                self.greedy_a[row][col], self.greedy_v[row][col] = action, self.Q[row][col][action]

    def choose_action(self, state):
        return self.greedy_a[state.agent_position[0]][state.agent_position[1]]
