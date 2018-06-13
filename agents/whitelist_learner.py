import operator
import numpy as np
from collections import defaultdict
from functools import partial
from random import choice as rchoice
from scipy import stats
from statsmodels.distributions.empirical_distribution import ECDF

from .q_learner import QLearner


# From https://stackoverflow.com/questions/36894191/how-to-get-a-normal-distribution-within-a-range-in-numpy
def get_truncated_normal(mean, sd, lower=0, upper=1, samples=100):
    sd = max(sd, 1e-10)  # make sure we can't divide by 0
    return stats.truncnorm((lower - mean) / sd, (upper - mean) / sd, loc=mean, scale=sd).rvs(samples)


class WhitelistLearner(QLearner):
    """A cautious agent that tries not to change the world too much in unknown ways."""
    unknown_cost = 150  # cost of each unknown change effected to the environment
    noise_restarts, noise_time_steps = 10, 10  # how many times to run environment for how many time steps

    def __init__(self, simulator, whitelist=set([]), sd=.05, mean=.8, do_train=True):
        # Prepare faux recognition - toggle confidence, noise
        self.recognition_samples = get_truncated_normal(mean, sd)

        # Generate second-best recognition candidates - should be roughly same each time a given object is recognized
        self.set_second_choices(simulator)
        self.whitelist = whitelist  # reuse the whitelist if we can

        # Inv-Gamma prior says smaller variances are somewhat more likely
        self.alpha, self.beta = 3, .5
        self.prior = ECDF(stats.t.rvs(df=2 * self.alpha, loc=0, scale=np.sqrt(self.beta / self.alpha), size=10000))
        self.posterior = defaultdict()
        self.dist = defaultdict()

        if do_train:
            self.set_noise(simulator)  # deduce noise for this simulator environment
            super().__init__(simulator)

    def set_second_choices(self, simulator):
        """Set second choices for mock classification."""
        objects = tuple(simulator.chars.values())
        self.other_objects = {key: self.get_other(key, objects) for key in objects}  # TODO expand to four

    def get_whitelist(self, examples, whitelist=set([])):
        """Extend / create a whitelist by classifying observed transitions in examples (lists of percept histories)."""
        for example in examples:
            old_state = self.observe_state(example[0]).flatten()
            for state_b in example[1:]:  # for each of the t-1 transitions
                new_state = self.observe_state(state_b).flatten()
                for old_dist, new_dist in zip(old_state, new_state):  # classify each square
                    sq, sq_prime = self.classify(old_dist), self.classify(new_dist)
                    if sq != sq_prime:
                        whitelist.add((sq, sq_prime))  # make assumption that classifier right - human can verify
                old_state = new_state
        return whitelist

    def recognize(self, obj, dynamic=False):
        """Simulate probabilistic object recognition. If dynamic enabled, randomly generate second-place candidate."""
        distribution = {obj: rchoice(self.recognition_samples)}
        distribution[self.get_other(obj, tuple(self.other_objects.keys())) if dynamic
                     else self.other_objects[obj]] = 1 - distribution[obj]
        return distribution

    def observe_state(self, state, dynamic=False):
        """Uncertain state recognition."""
        return np.array([[self.recognize(obj, dynamic) for obj in row] for row in state])

    @staticmethod
    def classify(dist):
        """Classify a probability distribution."""
        return max(dist.items(), key=operator.itemgetter(1))[0]

    def transition_shifts(self, old, new):
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
        noise = defaultdict(list)
        for _ in range(self.noise_restarts):
            for t in range(self.noise_time_steps):
                state = self.observe_state(simulator.state, dynamic=True).flatten()
                for obj_ind, dist in enumerate(state):
                    for key in dist.keys():
                        noise[obj_ind, key].append(dist[key])
                simulator.take_action(None)
            simulator.reset()

        # Merge noise for values across object identities (e.g. more than one vase - merge noise for both vases)
        merged_noise = defaultdict(partial(np.ndarray, 1))
        for key in noise.keys():
            converted = np.array(noise[key])
            merged_noise[key[1]] = np.append(merged_noise[key[1]], converted - converted.mean())
        for key in merged_noise.keys():
            n = len(merged_noise[key])
            # One n to get back to sum of squared errors, another for sample var correction
            self.posterior[key] = self.alpha + n/2, self.beta + n * n * merged_noise[key].var()/(2*(n-1))
        self.update_dist()

    def update_dist(self):
        """Regenerate the cached empirical distributions based on self.posterior (assumes all keys present)."""
        for key, (alpha, beta) in self.posterior.items():
            self.dist[key] = ECDF(stats.t.rvs(df=2 * alpha, loc=0, scale=np.sqrt(beta / alpha), size=1000))

    def get_noise(self, sq, sq_prime, shift):
        """Using learned noise distributions, return P(shift is not noise)."""
        shift_prime = shift/np.sqrt(2)
        return ((self.dist[sq](shift_prime) - .5) / .5) * ((self.dist[sq_prime](shift_prime) - .5) / .5)  # normalize on positive density

    def total_penalty(self, state_a, state_b):
        """Calculate the penalty incurred by the transition from state_a to state_b."""
        return sum([self.penalty(sq, sq_prime, shift)
                    for dist_a, dist_b in zip(state_a.flatten(), state_b.flatten())
                    for sq, sq_prime, shift in self.transition_shifts(dist_a, dist_b)])

    def penalty(self, sq, sq_prime, shift):
        """Using the whitelist average probability shifts, calculate penalty."""
        if sq == sq_prime or (sq, sq_prime) in self.whitelist: return 0
        prob = self.get_noise(sq, sq_prime, shift)  # compensate for observational noise in this specific environment
        return prob**5 * shift * self.unknown_cost

    def __str__(self):
        return "Whitelist"
