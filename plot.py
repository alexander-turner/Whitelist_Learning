import matplotlib.pyplot as plt

from agents.q_learner import QLearner
from agents.whitelist_learner import WhitelistLearner
from runner import MyCounter, run

"""Framework for collecting and plotting data for different observational noise levels."""
if __name__ == '__main__':
    num_levels = 0
    standard_deviations = (0, .001, .01, .025, .05, .075, .1, .125, .15, .175)
    q_data = (98, 0)  # already ran this experiment
    whitelist_data = {}

    for sd in standard_deviations:
        broken, failed, round = MyCounter(), MyCounter([QLearner, WhitelistLearner]), MyCounter()
        failed[QLearner], failed[WhitelistLearner] = 0, 0

        while round[0] < num_levels:  # random showcase
            run(use_q_learner=False, do_render=False, sd=sd)
        whitelist_data[sd] = broken[WhitelistLearner], failed[WhitelistLearner]

    print(whitelist_data)
    _, ax = plt.subplots()
    ax.set_title('Objects Broken vs. Observational Noise')
    ax.set_xlabel('Observational Noise')
    ax.set_ylabel('Objects Broken')
    ax.plot(standard_deviations, [q_data[0] for _ in standard_deviations], color='maroon', label='Q')
    ax.plot(standard_deviations, [whitelist_data[sd][1] for sd in standard_deviations], color='deepskyblue', label='Whitelist')
    ax.legend(loc='upper right')

    _, ax = plt.subplots()
    ax.set_title('Levels Failed vs. Observational Noise')
    ax.set_xlabel('Observational Noise')
    ax.set_ylabel('Levels Failed')
    ax.plot(standard_deviations, [q_data[1] for _ in standard_deviations], color='maroon', label='Q')
    ax.plot(standard_deviations, [whitelist_data[sd][1] for sd in standard_deviations], color='deepskyblue', label='Whitelist')
    ax.legend(loc='upper right')

    plt.show()
