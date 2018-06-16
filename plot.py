import matplotlib.pyplot as plt
import numpy as np

from agents.q_learner import QLearner
from agents.whitelist_learner import WhitelistLearner
from environments.vase_world.vases import VaseWorld
from runner import MyCounter, run


"""Framework for collecting and plotting data for different noise levels."""
if __name__ == '__main__':
    _, ax = plt.subplots()

    ax.set_title('Shift CDFs (actual noise σ=.05)')
    ax.set_ylabel('P(not noise)')
    ax.set_xlabel("Shift")

    agent = WhitelistLearner(VaseWorld(), sd=.05)
    def get_label(items, item_val):
        for name, val in items:
            if val == item_val: return name

    shifts = np.linspace(0, 1)
    for key, ECDF in agent.dist.items():
        print(key, agent.posterior[key][1] / agent.posterior[key][0])  # var
        y = (ECDF(shifts/np.sqrt(2)) - .5) / .5  # normalize
        ax.plot(shifts, y, label=get_label(VaseWorld.chars.items(), key))
    y = (agent.prior(shifts/np.sqrt(2)) - .5) / .5  # prior
    ax.plot(shifts, y, label='prior')

    ax.legend(loc='lower right')
    plt.savefig('post_noise.eps', format='eps', dpi=1000)
    plt.show()

    num_levels = 100
    standard_deviations = (0, .001, .01, .025, .05, .075, .1, .125, .15, .175)
    q_data = None
    whitelist_data = {}

    for sd in standard_deviations:
        print("Noise level: {}".format(sd))
        broken, failed, round = MyCounter(), MyCounter([QLearner, WhitelistLearner]), MyCounter()
        failed[QLearner], failed[WhitelistLearner] = 0, 0

        while round[0] < num_levels:  # TODO parallelize
            run(broken, failed, round, use_q_learner=sd == 0, do_render=False, sd=sd)
        if sd == 0:
            q_data = broken[QLearner], failed[QLearner]  # only generate for first noise condition
        whitelist_data[sd] = broken[WhitelistLearner], failed[WhitelistLearner]
        print()

    def set_up_plots():
        _, ax = plt.subplots()
        ax.set_xlabel('Noise (1σ)')
        ax.set_ylim(-.06 * num_levels, 1.06 * num_levels)
        return ax

    ax = set_up_plots()
    ax.set_title('Objects Broken vs. Noise (100 levels)')
    ax.set_ylabel('Objects Broken')
    ax.plot(standard_deviations, [q_data[0] for _ in standard_deviations], color='maroon', label='Q')
    ax.plot(standard_deviations, [whitelist_data[sd][0] for sd in standard_deviations], color='deepskyblue', label='Whitelist')
    ax.legend(loc='upper right')
    plt.savefig('broken.eps', format='eps', dpi=1000)

    ax = set_up_plots()
    ax.set_title('Levels Failed vs. Noise (100 levels)')
    ax.set_ylabel('Levels Failed')
    ax.plot(standard_deviations, [q_data[1] for _ in standard_deviations], color='maroon', label='Q')
    ax.plot(standard_deviations, [whitelist_data[sd][1] for sd in standard_deviations], color='deepskyblue', label='Whitelist')
    ax.legend(loc='upper right')
    plt.savefig('failed.eps', format='eps', dpi=1000)
    plt.show()
