from agents.q_learner import QLearner
from agents.whitelist_learner import WhitelistLearner
from environments.vases import VaseWorld

while True:
    state = VaseWorld(4, 4, .3)
    q, whitelist = QLearner(state), WhitelistLearner([[('A_', '_'), ('_', 'A_'), ('G', 'AG')]], state)
    for agent in (q, whitelist):
        while not state.is_terminal():
            state.take_action(agent.choose_action(state))
            print(state)
        state.reset()
        print('RESET')
