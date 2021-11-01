"""
Takes advantage of multicore systems to speed up the simulation runs.
"""
import matplotlib
matplotlib.use('qt4agg')
import sys

# change this path
sys.path.append("/Users/mxgo/classes/cs282br/bandits/bandits")

from agent import Agent, BetaAgent, TestAgent
from bandit import BernoulliBandit, BinomialBandit
from policy import GreedyPolicy, EpsilonGreedyPolicy, UCBPolicy
from environment import Environment, budget
# from environment_original import Environment

class BernoulliExample:
    label = 'Bayesian Bandits - Bernoulli'
    bandit = BernoulliBandit(10, t=3*10000)
    agents = [
        Agent(bandit, EpsilonGreedyPolicy(0.1)),
        Agent(bandit, UCBPolicy(1)),
        BetaAgent(bandit, GreedyPolicy()),
        TestAgent(bandit, budget=budget, c=1)
    ]

class BinomialExample:
    label = 'Bayesian Bandits - Binomial (n=5)'
    bandit = BinomialBandit(10, n=5, t=3*1000)
    agents = [
        Agent(bandit, EpsilonGreedyPolicy(0.1)),
        Agent(bandit, UCBPolicy(1)),
        BetaAgent(bandit, GreedyPolicy()),
        TestAgent(bandit, budget=budget)
    ]


if __name__ == '__main__':
    experiments = 100
    trials = 1000

    example = BernoulliExample()
    # example = BinomialExample()

    savefig="plots/bernoulli_budgets_03-07-100e-1000t"
    env = Environment(example.bandit, example.agents, example.label)
    scores, optimal, budgets, survival_rates = env.run(trials, experiments, budget=budget)
    env.plot_results(scores, optimal, budgets, survival_rates, savefig)
    env.plot_beliefs()
