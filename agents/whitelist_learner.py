import operator
from collections import defaultdict
from random import choice as rchoice

import numpy as np
from scipy.stats import truncnorm

from .q_learner import QLearner


# From https://stackoverflow.com/questions/36894191/how-to-get-a-normal-distribution-within-a-range-in-numpy
def get_truncated_normal(mean, sd, lower=0, upper=1, samples=50):
    sd = max(sd, 1e-10)  # make sure we can't divide by 0
    return truncnorm((lower - mean) / sd, (upper - mean) / sd, loc=mean, scale=sd).rvs(samples)


class WhitelistLearner(QLearner):
    """A cautious agent that tries not to change the world too much in unknown ways."""
    recognition_samples = get_truncated_normal(mean=.7, sd=.01)  # prepare faux recognition
    unknown_cost = 150  # cost of each unknown change effected to the environment
    noise_restarts, noise_time_steps = 10, 10  # how many times to run environment for how many time steps

    def __init__(self, simulator, examples=None, whitelist=None):
        """Takes a series of state representations (training set) and a simulator."""
        # Generate second-best recognition candidates - should be roughly same each time a given object is recognized
        self.set_second_choices(simulator)

        # Reuse the whitelist if we can
        self.whitelist = set([])  # TODO refactor
        if examples is not None and not whitelist:
            self.expand_whitelist(examples)
        else:
            self.whitelist = whitelist

        # Deduce noise for this simulator environment
        self.set_noise(simulator)
        super().__init__(simulator)  # do normal training

    def set_second_choices(self, simulator):
        """Set second choices for mock classification."""
        objects = tuple(simulator.chars.values())
        self.other_objects = {key: self.get_other(key, objects) for key in objects}

    def expand_whitelist(self, examples):
        for example in examples:
            old_state = self.observe_state(example[0]).flatten()
            for state_b in example[1:]:  # for each of the t-1 transitions
                new_state = self.observe_state(state_b).flatten()
                for old_dist, new_dist in zip(old_state, new_state):  # classify each square
                    sq, sq_prime = self.classify(old_dist), self.classify(new_dist)
                    if sq != sq_prime:
                        self.whitelist.add((sq, sq_prime))
                old_state = new_state

    def recognize(self, obj):
        """Simulate probabilistic object recognition. If dynamic enabled, randomly generate second-place candidate."""
        distribution = {obj: rchoice(self.recognition_samples)}
        distribution[self.other_objects[obj]] = 1 - distribution[obj]
        return distribution

    def observe_state(self, state):
        """Uncertain state recognition."""
        return np.array([[self.recognize(obj) for obj in row] for row in state])

    @staticmethod
    def classify(dist):
        """Classify a probability distribution."""
        return max(dist.items(), key=operator.itemgetter(1))[0]

    def transition_probabilities(self, old, new):
        """Get the possible probability shifts between the distributions."""
        increases, decreases, total_change = self.group_deltas(old, new)

        return [(sq, sq_prime, decrease * increase / total_change)
                for sq, decrease in decreases.items()  # how much probability mass did sq lose?
                for sq_prime, increase in increases.items()]  # and where could it have gone?

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

    def set_noise(self, simulator):
        """Learn environment-specific noise distributions at rest (\pi_{null} or \pi_{safe})."""
        self.noise = defaultdict(list)
        for _ in range(self.noise_restarts):
            old_state = self.observe_state(simulator.state).flatten()
            for t in range(self.noise_time_steps):
                simulator.take_action('rest')
                new_state = self.observe_state(simulator.state).flatten()
                for old_dist, new_dist in zip(old_state, new_state):  # for each square's distributions
                    for sq, sq_prime, probability in self.transition_probabilities(old_dist, new_dist):
                        self.noise[sq, sq_prime].append(probability)
                old_state = new_state
            simulator.reset()

        for key in self.noise.keys():
            converted = np.array(self.noise[key])
            self.noise[key] = converted.mean(), converted.std()

    def total_penalty(self, state_a, state_b):
        """Calculate the penalty incurred by the transition from state_a to state_b."""
        
        return sum([self.penalty(sq, sq_prime, probability)
                    for dist_a, dist_b in zip(state_a.flatten(), state_b.flatten())
                    for sq, sq_prime, probability in self.transition_probabilities(dist_a, dist_b)])

    def penalty(self, sq, sq_prime, shift):
        """Using the whitelist average probability shifts, calculate penalty."""
        if sq == sq_prime or (sq, sq_prime) in self.whitelist: return 0
        if (sq, sq_prime) not in self.noise: return shift * self.unknown_cost

        # Compensate for observational noise in this specific environment
        dist_above_mean = shift - self.noise[sq, sq_prime][0]
        std_dev = dist_above_mean / max(self.noise[sq, sq_prime][1], 1e-10)  # preclude division by 0
        activation = np.tanh(2*(std_dev - 1.5))  # filter out observational noise, punish outliers steeply
        return max(0, activation) * shift * self.unknown_cost

    def __str__(self):
        return "Whitelist"
