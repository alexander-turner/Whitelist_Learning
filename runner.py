import time
from collections import Counter
from random import random, randint

from agents.q_learner import QLearner
from agents.whitelist_learner import WhitelistLearner
from environments.vase_world.challenges import challenges
from environments.vase_world.vases import VaseWorld

examples = [[[['A_', '_']],  # 2 time steps of a 2x1 VaseWorld is all the whitelist learner needs
             [['_', 'A_']]]]

broken = Counter()


def run(state, level):
    """Run the given VaseWorld state for both learners."""
    for agent in (QLearner(state), WhitelistLearner(examples, state)):
        state.is_whitelist = isinstance(agent, WhitelistLearner)
        state.render()

        # shouldn't take more than w*h steps to complete; ensure whitelist isn't stuck behind obstacles
        while state.time_step < state.width * state.height and not state.is_terminal():
            time.sleep(.1)
            state.take_action(agent.choose_action(state))
            state.render()

        broken[agent.__class__] += sum([sum([1 for c in row if c == state.chars['mess']]) for row in state.state])
        if not isinstance(agent, WhitelistLearner):  # don't sleep if we're about to train
            time.sleep(.5)
        state.reset()
    level += 1  # how many levels have we ran?
    print('\rRound {}. {}'.format(level, broken), end='', flush=True)
    return level


level = 0
for challenge in challenges:  # curated showcase
    level = run(VaseWorld(state=challenge), level)

while True:  # random showcase
    level = run(VaseWorld(width=randint(4, 5), height=randint(4, 5), obstacle_chance=(random() + 1)/4),  # [0.25, 0.5]
                level)
