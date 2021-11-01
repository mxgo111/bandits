import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import scipy.stats as stats

from agent import BetaAgent

budget = 10

class Environment(object):
    def __init__(self, bandit, agents, label='Multi-Armed Bandit'):
        self.bandit = bandit
        self.agents = agents
        self.label = label

    def reset(self):
        self.bandit.reset()
        for agent in self.agents:
            agent.reset()

    def run(self, trials=100, experiments=1, budget=budget):
        scores = np.zeros((trials, len(self.agents)))
        print(scores.shape)
        total_budget = np.ones((trials, len(self.agents))) * budget
        total_budget_experiments = np.zeros(total_budget.shape)
        survival_rates = np.zeros(total_budget.shape)
        optimal = np.zeros_like(scores)

        for _ in range(experiments):
            self.reset()
            for t in range(trials):
                for i, agent in enumerate(self.agents):
                    if total_budget[t,i] <= 0:
                        if t < trials - 1:
                            total_budget[t+1,i] = total_budget[t,i]
                        # print(f"Agent {i} has already died")
                        continue
                    survival_rates[t,i] += 1
                    action = agent.choose()
                    reward, is_optimal = self.bandit.pull(action)
                    agent.observe(reward)

                    reward = reward * 2 - 1 # for -1 and +1 rewards

                    if t < trials - 1:
                        total_budget[t+1, i] = total_budget[t, i] + reward
                    total_budget_experiments[t,i] += total_budget[t,i]

                    scores[t, i] += reward
                    if is_optimal:
                        optimal[t, i] += 1

        return scores / experiments, optimal / experiments, total_budget_experiments / experiments, survival_rates / experiments

    def plot_results(self, scores, optimal, budgets, survival_rates, savefig):
        plt.figure(figsize=(30, 40))
        sns.set_style('white')
        sns.set_context('talk')
        plt.subplot(4, 1, 1)
        plt.title(self.label)
        plt.plot(scores)
        plt.ylabel('Average Reward')
        plt.legend(self.agents, loc=4)
        plt.subplot(4, 1, 2)
        plt.plot(optimal * 100)
        # plt.ylim(0, 100)
        plt.ylabel('% Optimal Action')
        plt.xlabel('Time Step')
        plt.legend(self.agents, loc=4)
        plt.subplot(4, 1, 3)
        plt.plot(budgets)
        # plt.ylim(0, 10)
        plt.ylabel('Remaining Budget')
        plt.xlabel('Time Step')
        plt.legend(self.agents, loc=4)
        plt.subplot(4, 1, 4)
        plt.plot(survival_rates)
        plt.ylabel('Survival Rate')
        plt.xlabel('Time Step')
        plt.legend(self.agents, loc=4)
        sns.despine()
        plt.savefig(savefig)
        plt.show()

    def plot_beliefs(self):
        sns.set_context('talk')
        pal = sns.color_palette("cubehelix", n_colors=len(self.agents))
        plt.title(self.label + ' - Agent Beliefs')

        rows = 2
        cols = int(self.bandit.k / 2)

        axes = [plt.subplot(rows, cols, i+1) for i in range(self.bandit.k)]
        for i, val in enumerate(self.bandit.action_values):
            color = 'r' if i == self.bandit.optimal else 'k'
            axes[i].vlines(val, 0, 1, colors=color)

        for i, agent in enumerate(self.agents):
            if type(agent) is not BetaAgent:
                for j, val in enumerate(agent.value_estimates):
                    axes[j].vlines(val, 0, 0.75, colors=pal[i], alpha=0.8)
            else:
                x = np.arange(0, 1, 0.001)
                y = np.array([stats.beta.pdf(x, a, b) for a, b in
                             zip(agent.alpha, agent.beta)])
                y /= np.max(y)
                for j, _y in enumerate(y):
                    axes[j].plot(x, _y, color=pal[i], alpha=0.8)

        min_p = np.argmin(self.bandit.action_values)
        for i, ax in enumerate(axes):
            ax.set_xlim(0, 1)
            if i % cols != 0:
                ax.set_yticklabels([])
            if i < cols:
                ax.set_xticklabels([])
            else:
                ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
                ax.set_xticklabels(['0', '', '0.5', '', '1'])
            if i == int(cols/2):
                title = '{}-arm Bandit - Agent Estimators'.format(self.bandit.k)
                ax.set_title(title)
            if i == min_p:
                ax.legend(self.agents)

        sns.despine()
        plt.show()
