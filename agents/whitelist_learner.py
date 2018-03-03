from collections import Counter, defaultdict
from random import choice as rchoice
from random import randint

import numpy as np

from .q_learner import QLearner


class WhitelistLearner(QLearner):
    """A cautious agent that tries not to change the world too much in unknown ways."""
    unknown_cost = 300  # cost of each unknown change effected to the environment

    def __init__(self, simulator, examples):
        """Takes a series of state representations (training set) and a simulator."""
        self.recognition = {obj: self.recognize(obj, tuple(simulator.chars.values()))
                            for obj in simulator.chars.values()}
        counts = Counter()
        self.whitelist = defaultdict(float)  # whitelist[sq1, sq2] := average prob shift for sq1 -> sq2 during training
        for example in examples:
            old_state = self.observe_state(example[0]).flatten()
            for state_b in example[1:]:  # for each of the t-1 transitory time slices
                new_state = self.observe_state(state_b).flatten()
                for old_dist, new_dist in zip(old_state, new_state):  # for each square's distributions
                    for sq, sq_prime, probability in self.transition_probabilities(old_dist, new_dist):
                        self.whitelist[sq, sq_prime] += probability
                        counts[sq, sq_prime] += 1
                old_state = new_state
        for key in self.whitelist.keys():
            self.whitelist[key] /= counts[key]
        super().__init__(simulator)  # do normal training

    def recognize(self, obj, objects):
        """Simulate probabilistic object recognition."""
        distribution = {obj: np.random.normal(.9, .001)}
        other_obj = rchoice(objects)  # make sure we don't select the same object twice
        while other_obj == obj:
            other_obj = rchoice(objects)
        distribution[other_obj] = 1 - distribution[obj]
        return distribution

    def observe_state(self, state):
        """Uncertain state recognition."""
        return np.array([[self.recognition[obj] for obj in row] for row in state])

    def transition_probabilities(self, old, new):
        """Get the possible probability shifts between the distributions."""
        increases, decreases, total_change = self.group_deltas(old, new)

        transitions = []
        for sq, decrease in decreases.items():  # how much probability mass did sq lose?
            for sq_prime, increase in increases.items():  # and where could it have gone?
                transitions.append((sq, sq_prime, decrease * increase / total_change))
        return transitions

    @staticmethod
    def group_deltas(old, new):
        """Return changes in probability mass (both the increases and the decreases)."""
        increases, decreases, total_change = {}, {}, 0
        for key, val in old.items():
            if key not in new:  # probability mass totally transferred
                decreases[key] = val
                total_change += val
            elif new[key] < old[key]:
                decreases[key] = old[key] - new[key]
                total_change += val  # only for decreases so we don't double-count
            elif new[key] > old[key]:
                increases[key] = new[key] - old[key]
        for key, val in new.items():
            if key not in old:
                increases[key] = val
        return increases, decreases, total_change

    def total_penalty(self, state_a, state_b):
        """Calculate the penalty incurred by the transition from state_a to state_b."""
        return sum([self.penalty(sq, sq_prime, probability) for dist_a, dist_b in zip(state_a.flatten(), state_b.flatten())
                    for sq, sq_prime, probability in self.transition_probabilities(dist_a, dist_b)])

    def penalty(self, square_a, square_b, shift=0):
        """Using the whitelist average probability shifts, calculate penalty."""
        if square_a == square_b: return 0
        if (square_a, square_b) not in self.whitelist: return self.unknown_cost
        return max(0, shift - self.whitelist[square_a, square_b]) * self.unknown_cost  # TODO more sophisticated?

    def train(self, simulator):
        while self.num_samples.min() < self.convergence_bound:
            new_pos = randint(0, simulator.height - 1), randint(0, simulator.width - 1)

            # Go to new simulator state and take action
            simulator.reset()
            simulator.set_agent_pos(simulator.agent_pos, np.array(new_pos))
            reward = simulator.get_reward()  # reward in state[row][col]
            old_state = self.observe_state(simulator.state)

            action = self.e_greedy_action(new_pos)  # choose according to explore/exploit
            simulator.take_action(self.actions[action])
            self.num_samples[new_pos][action] += 1  # update sample count

            new_state = self.observe_state(simulator.state)
            self.update_greedy(new_pos, action, reward - self.total_penalty(old_state, new_state), simulator)
