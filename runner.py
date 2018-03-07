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

break_crates = False  # <== toggle me!
examples = np.array([[[['c']],
                      [['x']]]]) if break_crates else []
whitelist = WhitelistLearner(VaseWorld, do_train=False).get_whitelist(examples)


class MyCounter(Counter):
    """A print-friendly Counter class."""
    def __str__(self):
        return " ".join('{}: {}'.format(k.__str__(k), v) for k, v in self.items())


def initialize(agent, sim, training=None):
    """Multiprocessing-compliant agent initialization."""
    return agent(sim, training) if training is not None else agent(sim)


def run(simulator, do_render=True):
    """Run the given VaseWorld state for both learners."""
    with multiprocessing.Pool(processes=min(2, multiprocessing.cpu_count() - 1)) as pool:
        agents = pool.starmap(initialize, ([QLearner, simulator, None],
                                           [WhitelistLearner, deepcopy(simulator), whitelist]))
    for agent in agents:
        simulator.is_whitelist = isinstance(agent, WhitelistLearner)

        # Shouldn't take more than w*h steps to complete; ensure whitelist isn't stuck behind obstacles
        while simulator.time_step < simulator.num_squares and not simulator.is_terminal():
            if do_render:
                time.sleep(.1)
                simulator.render(agent.observe_state(simulator.state))
            simulator.take_action(agent.choose_action(simulator))

        if do_render:
            time.sleep(.1)
            simulator.render(agent.observe_state(simulator.state))

        broken[agent.__class__] += (simulator.state == simulator.chars['mess']).sum()
        if not simulator.is_terminal() and simulator.clearable:
            failed[agent.__class__] += 1
        if do_render and not simulator.is_whitelist:  # don't sleep if we're about to train
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
