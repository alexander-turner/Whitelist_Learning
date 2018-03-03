import time
from collections import Counter
from random import random, randint

import numpy as np

from agents.q_learner import QLearner
from agents.whitelist_learner import WhitelistLearner
from environments.vase_world.challenges import challenges
from environments.vase_world.vases import VaseWorld

examples = np.array([[[['_', '_']],  # 2 time steps of a 2x1 VaseWorld is all the whitelist learner needs TODO Clarify
                      [['_', '_']]]])

broken = Counter()
round = Counter()


def run(simulator):
    """Run the given VaseWorld state for both learners."""
    for agent in (QLearner(simulator), WhitelistLearner(simulator, examples)):
        simulator.is_whitelist = isinstance(agent, WhitelistLearner)
        simulator.render(agent.observe_state(simulator.state) if isinstance(agent, WhitelistLearner) else None)

        # Shouldn't take more than w*h steps to complete; ensure whitelist isn't stuck behind obstacles
        while simulator.time_step < simulator.width * simulator.height and not simulator.is_terminal():
            time.sleep(.1)
            simulator.take_action(agent.choose_action(simulator))
            simulator.render(agent.observe_state(simulator.state) if isinstance(agent, WhitelistLearner) else None)

        broken[agent.__class__] += (simulator.state == simulator.chars['mess']).sum()
        if not isinstance(agent, WhitelistLearner):  # don't sleep if we're about to train
            time.sleep(.5)
        simulator.reset()
    round[0] += 1  # how many levels have we ran?
    print('\rRound {}. {}'.format(round[0], broken), end='', flush=True)
    # TODO quantify completion percentage for levels which can be completed without breaking objects


for challenge in challenges:  # curated showcase
    run(VaseWorld(state=challenge))

for _ in range(1000):  # random showcase
    run(VaseWorld(height=randint(4, 5), width=randint(4, 5), obstacle_chance=(random() + 1) / 4))
