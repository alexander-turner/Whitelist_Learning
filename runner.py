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


def initialize(agent, sim, training=None, sd=.05):
    """Multiprocessing-compliant agent initialization."""
    return agent(sim, training, sd=sd) if training is not None else agent(sim)


def run(broken, failed, round, simulator=None, use_q_learner=False, do_render=True, sd=.025):
    """Run the given VaseWorld state for the specified learners."""
    if simulator is None:
        simulator = VaseWorld(height=randint(4, 5), width=randint(4, 5), obstacle_chance=(random() + 1) / 4)

    # Train the agents
    if use_q_learner:
        with multiprocessing.Pool(processes=min(2, multiprocessing.cpu_count() - 1)) as pool:
            agents = pool.starmap(initialize, ([QLearner, simulator],
                                               [WhitelistLearner, deepcopy(simulator), whitelist, sd]))
    else:
        agents = (WhitelistLearner(simulator, whitelist, sd=sd),)

    for agent in agents:
        simulator.is_whitelist = isinstance(agent, WhitelistLearner)

        # Shouldn't take more than w*h steps to complete - ensure whitelist isn't stuck behind obstacles
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
        run(broken, failed, round, simulator=VaseWorld(state=challenge))

    while round[0] < (1000 - len(challenges)):  # random showcase
        run(broken, failed, round)
