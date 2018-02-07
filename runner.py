import time
from random import random, randint

from agents.q_learner import QLearner
from agents.whitelist_learner import WhitelistLearner
from environments.vase_world.vases import VaseWorld


def run(state):
    """Run the given VaseWorld state for both learners."""
    for agent in (QLearner(state), WhitelistLearner(examples, state)):
        state.is_whitelist = isinstance(agent, WhitelistLearner)
        state.render()

        # shouldn't take more than w*h steps to complete; ensure whitelist isn't stuck behind obstacles
        while state.time_step < state.width * state.height and not state.is_terminal():
            time.sleep(.1)
            state.take_action(agent.choose_action(state))
            state.render()

        # Don't sleep if we're about to train
        if not isinstance(agent, WhitelistLearner):
            time.sleep(.5)
        state.reset()


examples = [[[['A_', '_']],  # 2 time steps of a 1x2 VaseWorld is all the whitelist learner needs
             [['_', 'A_']]]]

#for challenge in challenges:  # curated showcase
#    run(VaseWorld(state=challenge))

while True:  # random showcase
    run(VaseWorld(width=randint(3, 5), height=randint(3, 5),
                  obstacle_chance=(random() + 1)/4))  # in [0.25, 0.75]
