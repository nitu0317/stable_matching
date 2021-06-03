import numpy as np
from gurobipy import *
from collections import *
from itertools import chain, combinations
import cvxpy as cp
from graph import *
from opt import *
from estimate import *
import time

def recommendation(persist_value, men, women, leakage_arc):
    rank = {}
    for i in men[1:]:
        values = {j:persist_value[i, j] for j in women[1:] if (i, j) not in leakage_arc and persist_value[i, j] > 1e-6}
        rank[i] = sorted(values, key=values.get, reverse=True)

    return rank

def metric(tops, eval_samples, rank, men, women):
    for top_K in tops:
        recommand = {}
        for i in men[1:]:
            recommand[i] = rank[i][:min(top_K, len(rank[i]))]

        enumerates = np.concatenate([recommand[i] for i in men[1:]])

        ## cover rate
        cover = len(np.unique(enumerates))

        ## dist count
        distr = []
        for j in women[1:]:
            distr.append(np.count_nonzero(enumerates == j))

        ## pred accuracy
        acc = 0

        for eval_sample in eval_samples:
            for (i, j) in eval_sample:
                if j in recommand[i]:
                    acc += 1 / (100 * (len(men)-1))

    return acc, distr, cover

def evaluation(T, match_opt, men, women, eta, utility):
    eval_samples = []
    np.random.seed(1234)
    for n in range(T):
        utility_random = {}
        for i in men:
            for j in women:
                epsilon = np.random.gumbel(0, eta)
                utility_random[i, j]  = utility[i, j] + epsilon

                epsilon = np.random.gumbel(0, eta)
                utility_random[j, i] = utility[j, i] + epsilon

        model, x = match_opt.matching(True, utility_random)
        res_x = model.getAttr('X', x)
        pairs = []
        for (i, j), value in res_x.items():
            if value > 0.9 and i != 'mNA':
                pairs.append((i, j))

        eval_samples.append(pairs)

    return eval_samples

def main():
    eta = 0.5 # scale
    regime = (2, 4) # intrinsic
    N_men = 10 # freelancer
    N_women = 10 # jobs
    Y = 100 # instances
    M = 32 # sparse scenarios
    gamma = 0.2 # star proportion
    T = 100 # evaluation realizations
    tops = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] # top recommendations
    seed = 0

    ## setup models
    men, women, utility = GRAPH.customer_setup(seed, N_men, N_women, regime, gamma)
    leakage_arc = {('mNA', 'wNA')}
    estimate = ESTIMATE(men, women, leakage_arc, eta, utility)
    match_opt = MATCH_OPT(men, women, leakage_arc, eta, utility)

    ## estimate persistency
    persisent_value = estimate.SAA(seed, Y)

    ## createe testing samples
    eval_samples = evaluation(T, match_opt, men, women, eta, utility)

    ## generate rank
    rank = recommendation(persisent_value, men, women, leakage_arc)

    ## output metrics
    accuracy, distr_count, coverage = metric(tops, eval_samples, rank, men, women)

    print(accuracy)

    print(distr_count)

    print(coverage)

if __name__ == '__main__':
	main()
