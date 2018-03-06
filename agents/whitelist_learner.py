from collections import defaultdict
from random import choice as rchoice
from random import randint

import numpy as np
from scipy.stats import truncnorm

from .q_learner import QLearner


# From https://stackoverflow.com/questions/36894191/how-to-get-a-normal-distribution-within-a-range-in-numpy
def get_truncated_normal(mean, sd, lower=0, upper=1):
    return truncnorm((lower - mean) / sd, (upper - mean) / sd, loc=mean, scale=sd)


class WhitelistLearner(QLearner):
    """A cautious agent that tries not to change the world too much in unknown ways."""
    recognition_samples = get_truncated_normal(mean=.7, sd=.05).rvs(50)  # prepare faux recognition
    unknown_cost = 150  # cost of each unknown change effected to the environment

    def __init__(self, simulator, examples):
        """Takes a series of state representations (training set) and a simulator."""
        # Generate second-best recognition candidates - should be roughly same each time a given object is recognized
        objects = tuple(simulator.chars.values())
        self.other_objects = {key: self.get_other(key, objects) for key in objects}

        # whitelist[sq, sq_prime] := average prob shift for sq1 -> sq2 during training
        self.whitelist = defaultdict(list)
        for example in examples:
            old_state = self.observe_state(example[0], dynamic=True).flatten()
            for state_b in example[1:]:  # for each of the t-1 transitions
                new_state = self.observe_state(state_b, dynamic=True).flatten()
                for old_dist, new_dist in zip(old_state, new_state):  # for each square's distributions
                    for sq, sq_prime, probability in self.transition_probabilities(old_dist, new_dist):
                        self.whitelist[sq, sq_prime].append(probability)
                old_state = new_state
        for key in self.whitelist.keys():
            converted = np.array(self.whitelist[key])
            self.whitelist[key] = converted.mean(), converted.std(), len(converted)
        super().__init__(simulator)  # do normal training

    def recognize(self, obj, dynamic=False):
        """Simulate probabilistic object recognition. If dynamic enabled, randomly generate second-place candidate."""
        distribution = {obj: rchoice(self.recognition_samples)}
        distribution[self.get_other(obj, tuple(self.other_objects.keys())) if dynamic
                     else self.other_objects[obj]] = 1 - distribution[obj]
        return distribution

    def observe_state(self, state, dynamic=False):
        """Uncertain state recognition."""
        return np.array([[self.recognize(obj, dynamic) for obj in row] for row in state])

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
            if key not in old and val > 0:
                increases[key] = val
        return increases, decreases, total_change

    def total_penalty(self, state_a, state_b):
        """Calculate the penalty incurred by the transition from state_a to state_b."""
        return sum([self.penalty(sq, sq_prime, probability)
                    for dist_a, dist_b in zip(state_a.flatten(), state_b.flatten())
                    for sq, sq_prime, probability in self.transition_probabilities(dist_a, dist_b)])

    def penalty(self, sq, sq_prime, shift):
        """Using the whitelist average probability shifts, calculate penalty."""
        if sq == sq_prime: return 0
        if (sq, sq_prime) not in self.whitelist: return shift * self.unknown_cost
        # Calculate dissimilarity to distribution of training set transitions
        dist_above_mean = shift - self.whitelist[sq, sq_prime][0]
        std_dev = dist_above_mean / max(self.whitelist[sq, sq_prime][1], 1e-10)  # preclude division by 0
        activation = np.tanh(2*(std_dev - 1.5))  # filter out observational noise, punish outliers steeply
        return max(0, activation) * shift * self.unknown_cost

    def train(self, simulator):
        while self.num_samples.min() < self.convergence_bound:
            start_pos = randint(0, simulator.height - 1), randint(0, simulator.width - 1)
            if (start_pos == simulator.goal_pos).all():
                continue

            # Go to new simulator state and take action
            simulator.reset()
            simulator.agent_pos = np.array(start_pos)
            old_state = self.observe_state(simulator.state)

            action = self.e_greedy_action(start_pos)  # choose according to explore/exploit
            reward = simulator.take_action(self.actions[action])
            self.num_samples[action][start_pos] += 1  # update sample count

            new_state = self.observe_state(simulator.state)
            penalty = self.total_penalty(old_state, new_state)
            self.update_greedy(start_pos, action, reward - penalty, simulator)

    def __str__(self):
        return "Whitelist"
