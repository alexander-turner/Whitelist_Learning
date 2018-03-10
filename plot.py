import matplotlib.pyplot as plt

from agents.q_learner import QLearner
from agents.whitelist_learner import WhitelistLearner
from runner import MyCounter, run

"""Framework for collecting and plotting data for different noise levels."""
if __name__ == '__main__':
    num_levels = 100
    standard_deviations = (0, .001, .01, .025, .05, .075, .1, .125, .15, .175)
    q_data = None
    whitelist_data = {}

    for sd in standard_deviations:
        print("Noise level: {}".format(sd))
        broken, failed, round = MyCounter(), MyCounter([QLearner, WhitelistLearner]), MyCounter()
        failed[QLearner], failed[WhitelistLearner] = 0, 0

        while round[0] < num_levels:
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
