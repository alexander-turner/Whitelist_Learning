from random import randint
from environments.vases import VaseWorld

for _ in range(10):
    state = VaseWorld(4, 4)
    for _ in range(20):
        print(state.take_action(state.get_actions()[randint(0, 3)]))
        print(state)