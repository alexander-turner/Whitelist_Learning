import time

from agents.q_learner import QLearner
from agents.whitelist_learner import WhitelistLearner
from environments.vase_world.vases import VaseWorld

# This example encapsulates all the whitelist learner needs
examples = [[[['A_', '_', 'G']],  # 3 time steps of a 1x3 VaseWorld - look ma, no vase!
             [['_', 'A_', 'G']],
             [['_', '_', 'AG']]]]

while True:
    state = VaseWorld(4, 4, .2)
    q, whitelist = QLearner(state), WhitelistLearner(examples, state)
    for agent in (q, whitelist):
        state.is_whitelist = isinstance(agent, WhitelistLearner)
        state.render()
        while not state.is_terminal():
            time.sleep(.1)
            state.take_action(agent.choose_action(state))
            state.render()
        time.sleep(.3)
        state.reset()
