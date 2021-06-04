import numpy as np
from gurobipy import *
from collections import *
from itertools import chain, combinations
import cvxpy as cp
from graph import *
from opt import *
from estimate import *
import time
import argparse

def recommendation(persist_value, men, women, leakage_arc):
    rank = {}
    for i in men[1:]:
        values = {j:persist_value[i, j] for j in women[1:] if (i, j) not in leakage_arc and persist_value[i, j] > 1e-6}
        rank[i] = sorted(values, key=values.get, reverse=True)

    return rank

def metric(tops, eval_samples, rank, men, women):
    recommend_acc, recommend_distr, recommend_cover = [], [], []
    for top_K in tops:
        recommend = {}
        for i in men[1:]:
            recommend[i] = rank[i][:min(top_K, len(rank[i]))]

        enumerates = np.concatenate([recommend[i] for i in men[1:]])

        ## cover rate
        cover = len(np.unique(enumerates)) / len(women)
        recommend_cover.append(cover)

        ## dist count
        distr = []
        for j in women[1:]:
            distr.append(np.count_nonzero(enumerates == j))
        recommend_distr.append(distr)

        ## pred accuracy
        acc = 0

        for eval_sample in eval_samples:
            for (i, j) in eval_sample:
                if j in recommend[i]:
                    acc += 1 / (100 * (len(men)-1))
        recommend_acc.append(acc)

    return recommend_acc, recommend_distr, recommend_cover

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

def show_output(summary_acc, summary_distr, summary_cover, tops):
    ave_accuracy = np.round(np.mean(np.array(summary_acc), axis=0), 4)
    print("recommendation accuracy with top K: %s" % ave_accuracy)

    ave_coverage = np.round(np.mean(np.array(summary_cover), axis=0), 4)
    print("recommendation coverage with top K: %s" % ave_coverage)

    K = 4
    ave_min = np.mean(np.min(np.array(summary_distr)[:, K, :], axis=1))
    ave_max = np.mean(np.max(np.array(summary_distr)[:, K, :], axis=1))
    ave_mean = np.mean(np.mean(np.array(summary_distr)[:, K, :], axis=1))
    ave_median = np.mean(np.median(np.array(summary_distr)[:, K, :], axis=1))
    ave_std = np.mean(np.std(np.array(summary_distr)[:, K, :], axis=1))

    print("recommendation count distribution for top %s: min %s, max %s, median %s, mean %s, std %s." % (tops[K], ave_min, ave_max, ave_median, ave_mean, ave_std))

def main():
    parser = argparse.ArgumentParser(description='stable_matching')
    parser.add_argument('--eta', '-eta', help='scale', default=0.5)
    parser.add_argument('--regime', help='mu range', default=(2, 4))
    parser.add_argument('--N_men', '-f', help='number of freelancers', default=30)
    parser.add_argument('--N_women', '-j', help='number of jobs', default=30)
    parser.add_argument('--runs', '-runs', help='number of problem instances', default=50)
    parser.add_argument('--Y', '-Y', help='number of samples in each instance', default=500)
    parser.add_argument('--M', '-M', help='number of sparse scenarios', default=32)
    parser.add_argument('--T', '-T', help='number of evaluation realizations', default=100)
    parser.add_argument('--gamma', '-gamma', help='star proportion', default=0.2)
    parser.add_argument('--tops', '-tops', help='recommend sizes', default=np.arange(1, 11))
    parser.add_argument('--method', '-m', required=True)
    args = parser.parse_args()

    summary_acc, summary_distr, summary_cover = [], [], []
    for run in range(args.runs):
        print('Run %s' % run)
        seed = run

        ## setup models
        men, women, utility = GRAPH.customer_setup(seed, args.N_men, args.N_women, args.regime, args.gamma)
        leakage_arc = {('mNA', 'wNA')}
        estimate = ESTIMATE(men, women, leakage_arc, args.eta, utility)
        match_opt = MATCH_OPT(men, women, leakage_arc, args.eta, utility)

        ## estimate persistency
        if args.method == 'SAA':
            persist_value = estimate.SAA(seed, args.Y)
        elif args.method == 'ENT':
            persist_value = estimate.ENT()
        elif args.method == 'GRE':
            persist_value = estimate.GRE()
        elif args.method == 'SLP':
            persist_value = estimate.SLP(seed, args.Y, args.M)

        ## createe testing samples
        eval_samples = evaluation(args.T, match_opt, men, women, args.eta, utility)

        ## generate rank
        rank = recommendation(persist_value, men, women, leakage_arc)

        ## output metrics
        accuracy, distr_count, coverage = metric(args.tops, eval_samples, rank, men, women)

        summary_acc.append(accuracy)
        summary_distr.append(distr_count)
        summary_cover.append(coverage)

    ## print results
    show_output(summary_acc, summary_distr, summary_cover, args.tops)

if __name__ == '__main__':
	main()
