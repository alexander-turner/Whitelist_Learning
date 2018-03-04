import time
from collections import Counter
from random import random, randint

import numpy as np

from agents.q_learner import QLearner
from agents.whitelist_learner import WhitelistLearner
from environments.vase_world.challenges import challenges
from environments.vase_world.vases import VaseWorld

'''
Although the whitelist isn't necessary simple maze-navigation purposes, try swapping the comments - the learner will be 
okay breaking crates, but not vases! 
'''
#examples = np.array([[[['c', '_']],
#                      [['x', '_']]]])
examples = np.array([[[['_', '_']],
                      [['_', '_']]]])

broken, clearable, round = Counter(), Counter(), Counter()


def run(simulator):
    """Run the given VaseWorld state for both learners."""
    for agent in (QLearner(simulator), WhitelistLearner(simulator, examples)):
        simulator.is_whitelist = isinstance(agent, WhitelistLearner)
        simulator.render(agent.observe_state(simulator.state) if simulator.is_whitelist else None)

        # Shouldn't take more than w*h steps to complete; ensure whitelist isn't stuck behind obstacles
        while simulator.time_step < simulator.width * simulator.height and not simulator.is_terminal():
            time.sleep(.1)
            simulator.take_action(agent.choose_action(simulator))
            simulator.render(agent.observe_state(simulator.state) if simulator.is_whitelist else None)

        broken[agent.__class__] += (simulator.state == simulator.chars['mess']).sum()
        if not simulator.is_whitelist:  # don't sleep if we're about to train
            time.sleep(.5)
        simulator.reset()
    round[0] += 1  # how many levels have we ran?
    print('\rRound {}. {}'.format(round[0], broken), end='', flush=True)
    # TODO quantify completion percentage for levels which can be completed without breaking objects


for challenge in challenges:  # curated showcase
    run(VaseWorld(state=challenge))

for _ in range(1000):  # random showcase
    run(VaseWorld(height=randint(4, 5), width=randint(4, 5), obstacle_chance=(random() + 1) / 4))
