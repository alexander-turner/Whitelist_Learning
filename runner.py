import time

from agents.q_learner import QLearner
from agents.whitelist_learner import WhitelistLearner
from environments.vase_world.challenges import challenges
from environments.vase_world.vases import VaseWorld

# This example encapsulates all the whitelist learner needs
examples = [[[['A_', '_', 'G']],  # 3 time steps of a 1x3 VaseWorld
             [['_', 'A_', 'G']],
             [['_', '_', 'AG']]]
            ]

while True:
    for challenge in challenges:
        state = VaseWorld(state=challenge)
        for agent in (QLearner(state), WhitelistLearner(examples, state)):
            state.is_whitelist = isinstance(agent, WhitelistLearner)
            state.render()
            while not state.is_terminal():
                time.sleep(.1)
                state.take_action(agent.choose_action(state))
                state.render()

            # Don't sleep if we're about to train anyways
            if not isinstance(agent, WhitelistLearner):
                time.sleep(.5)
            state.reset()
