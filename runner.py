from agents.q_learner import QLearner
from environments.vases import VaseWorld

state = VaseWorld(4, 4)
agent = QLearner(state)
while True:
    while not state.is_terminal():
        state.take_action(agent.choose_action(state))
        print(state)
    state.reset()
    print('RESET')
