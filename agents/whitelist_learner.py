from collections import Counter, defaultdict
from random import choice as rchoice
from random import randint

import numpy as np

from .q_learner import QLearner


class WhitelistLearner(QLearner):
    """A cautious agent that tries not to change the world too much in unknown ways."""
    unknown_cost = 200  # cost of each unknown change effected to the environment

    def __init__(self, simulator, examples):  # TODO make it not retrain on examples each time?
        """Takes a series of state representations (training set) and a simulator."""
        # Prepare faux recognition
        self.recognition_samples = np.random.normal(.9, .05, 100)
        objects, self.other_objects = tuple(simulator.chars.values()), dict.fromkeys(simulator.chars.values())

        # Generate second-best recognition candidates - should be roughly same each time a given object is recognized
        for key in self.other_objects.keys():
            self.other_objects[key] = rchoice(objects)
            if len(objects) > 1:  # make sure there *is* another object we can choose
                while self.other_objects[key] == key:  # make sure we don't select the same object twice
                    self.other_objects[key] = rchoice(objects)

        # whitelist[sq, sq_prime] := average prob shift for sq1 -> sq2 during training
        counts, self.whitelist = Counter(), defaultdict(float)
        for example in examples:
            old_state = self.observe_state(example[0]).flatten()
            for state_b in example[1:]:  # for each of the t-1 transitions
                new_state = self.observe_state(state_b).flatten()
                for old_dist, new_dist in zip(old_state, new_state):  # for each square's distributions
                    for sq, sq_prime, probability in self.transition_probabilities(old_dist, new_dist):
                        self.whitelist[sq, sq_prime] += probability
                        counts[sq, sq_prime] += 1
                old_state = new_state
        for key in self.whitelist.keys():
            self.whitelist[key] /= counts[key]
        super().__init__(simulator)  # do normal training

    def recognize(self, obj):
        """Simulate probabilistic object recognition."""
        distribution = {obj: rchoice(self.recognition_samples)}
        distribution[self.other_objects[obj]] = 1 - distribution[obj]
        return distribution

    def observe_state(self, state):
        """Uncertain state recognition."""
        return np.array([[self.recognize(obj) for obj in row] for row in state])

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
            if key not in new or new[key] < old[key]:  # probability mass totally transferred
                decreases[key] = old[key] - (new[key] if key in new else 0)
                total_change += val  # only for decreases so we don't double-count
            elif new[key] > old[key]:
                increases[key] = new[key] - old[key]
        for key, val in new.items():
            if key not in old:
                increases[key] = val
        return increases, decreases, total_change

    def total_penalty(self, state_a, state_b):
        """Calculate the penalty incurred by the transition from state_a to state_b."""
        return sum([self.penalty(sq, sq_prime, probability)  # TODO background penalty
                    for dist_a, dist_b in zip(state_a.flatten(), state_b.flatten())
                    for sq, sq_prime, probability in self.transition_probabilities(dist_a, dist_b)])

    def penalty(self, square_a, square_b, shift):
        """Using the whitelist average probability shifts, calculate penalty."""
        if square_a == square_b: return 0
        if (square_a, square_b) not in self.whitelist: return shift * self.unknown_cost
        return max(0, shift - self.whitelist[square_a, square_b]) * self.unknown_cost  # TODO more sophisticated?

    def train(self, simulator):
        while self.num_samples.min() < self.convergence_bound:
            # Go to new simulator state and take action
            simulator.reset()
            start_pos = randint(0, simulator.height - 1), randint(0, simulator.width - 1)
            simulator.agent_pos = np.array(start_pos)
            old_state = self.observe_state(simulator.state)

            action = self.e_greedy_action(start_pos)  # choose according to explore/exploit
            reward = simulator.take_action(self.actions[action])
            self.num_samples[start_pos][action] += 1  # update sample count

            new_state = self.observe_state(simulator.state)
            self.update_greedy(start_pos, action, reward - self.total_penalty(old_state, new_state), simulator)
