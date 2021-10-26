"""
Takes advantage of multicore systems to speed up the simulation runs.
"""
import matplotlib
matplotlib.use('qt4agg')
import sys
sys.path.append("/Users/mxgo/classes/cs282br/bandits/bandits")

from agent import Agent, BetaAgent
from bandit import BernoulliBandit, BinomialBandit
from policy import GreedyPolicy, EpsilonGreedyPolicy, UCBPolicy
from environment import Environment

class BernoulliExample:
    label = 'Bayesian Bandits - Bernoulli'
    bandit = BernoulliBandit(10, t=3*1000)
    agents = [
        Agent(bandit, EpsilonGreedyPolicy(0.1)),
        Agent(bandit, UCBPolicy(1)),
        BetaAgent(bandit, GreedyPolicy())
    ]


class BinomialExample:
    label = 'Bayesian Bandits - Binomial (n=5)'
    bandit = BinomialBandit(10, n=5, t=3*1000)
    agents = [
        Agent(bandit, EpsilonGreedyPolicy(0.1)),
        Agent(bandit, UCBPolicy(1)),
        BetaAgent(bandit, GreedyPolicy())
    ]


if __name__ == '__main__':
    experiments = 50
    trials = 100

    example = BernoulliExample()
    # example = BinomialExample()

    env = Environment(example.bandit, example.agents, example.label)
    scores, optimal = env.run(trials, experiments)
    env.plot_results(scores, optimal)
    env.plot_beliefs()
