import multiprocessing
import time
from collections import Counter
from copy import deepcopy
from random import random, randint

import numpy as np

from agents.q_learner import QLearner
from agents.whitelist_learner import WhitelistLearner
from environments.vase_world.challenges import challenges
from environments.vase_world.vases import VaseWorld

'''
Try swapping the comments - the learner will be more comfortable breaking crates, but not vases! 
'''
#examples = np.array([[[['c', '_']],  # TODO fix
#                      [['x', '_']]]])
examples = np.array([[[['_']],
                      [['_']]]])

class MyCounter(Counter):
    """A print-friendly Counter class."""
    def __str__(self):
        return " ".join('{}: {}'.format(k.__str__(k), v) for k, v in self.items())


def initialize(agent, sim, training=None):
    """Multiprocessing-compliant agent initialization."""
    return agent(sim, training) if training is not None else agent(sim)


def run(simulator):
    """Run the given VaseWorld state for both learners."""
    with multiprocessing.Pool(processes=min(2, multiprocessing.cpu_count() - 1)) as pool:
        agents = pool.starmap(initialize, ([QLearner, simulator, None],
                                           [WhitelistLearner, deepcopy(simulator), examples]))
    for agent in agents:
        simulator.is_whitelist = isinstance(agent, WhitelistLearner)
        simulator.render(agent.observe_state(simulator.state) if simulator.is_whitelist else None)

        # Shouldn't take more than w*h steps to complete; ensure whitelist isn't stuck behind obstacles
        while simulator.time_step < simulator.num_squares and not simulator.is_terminal():
            time.sleep(.1)
            simulator.take_action(agent.choose_action(simulator))
            simulator.render(agent.observe_state(simulator.state) if simulator.is_whitelist else None)

        broken[agent.__class__] += (simulator.state == simulator.chars['mess']).sum()
        if not simulator.is_terminal() and simulator.clearable:
            failed[agent.__class__] += 1
        if not simulator.is_whitelist:  # don't sleep if we're about to train
            time.sleep(.5)
        simulator.reset()
    round[0] += 1  # how many levels have we ran?
    print('\rRound {} | Objects broken  {} | Levels failed  {}'.format(round[0], broken, failed), end='', flush=True)


if __name__ == '__main__':
    broken, failed, round = MyCounter(), MyCounter([QLearner, WhitelistLearner]), MyCounter()
    failed[QLearner], failed[WhitelistLearner] = 0, 0

    for challenge in challenges:  # curated showcase
        run(VaseWorld(state=challenge))

    while round[0] < 1000:  # random showcase
        run(VaseWorld(height=randint(4, 5), width=randint(4, 5), obstacle_chance=(random() + 1) / 4))
