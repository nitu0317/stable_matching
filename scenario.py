import numpy as np
from gurobipy import *
from collections import *
from itertools import chain, combinations
import cvxpy as cp
import time

class SCENARIO():
    def __init__(self, men, women, leakage_arc, eta, utility):
        self.men = men
        self.women = women
        self.leakage_arc = leakage_arc
        self.eta = eta
        self.utility = utility

    def preference_pair(self, seed, M, Y):
        total_men = {(i,j):defaultdict(list) for i in self.men[1:] for j in self.women if (i, j) not in self.leakage_arc}
        total_women = {(j,i):defaultdict(list) for i in self.men for j in self.women[1:] if (i, j) not in self.leakage_arc}
        count_men = {(i,j):defaultdict(float) for i in self.men[1:] for j in self.women if (i, j) not in self.leakage_arc}
        count_women = {(j,i):defaultdict(float) for i in self.men for j in self.women[1:] if (i, j) not in self.leakage_arc}

        np.random.seed(seed)
        for _ in range(Y):
            utility_random = {}
            for i in self.men:
                for j in self.women:
                    if (i, j) not in self.leakage_arc:
                        epsilon = np.random.gumbel(0, self.eta)
                        utility_random[i, j]  = self.utility[i, j] + epsilon

                        epsilon = np.random.gumbel(0, self.eta)
                        utility_random[j, i] = self.utility[j, i] + epsilon

            for i in self.men[1:]:
                for j in self.women:
                    if (i, j) not in self.leakage_arc:
                        large, small = [], []
                        for k in self.women:
                            if (i, k) not in self.leakage_arc and k != j:
                                if utility_random[i, k] > utility_random[i, j]:
                                    large.append(k)
                                else:
                                    small.append(k)

                                pair = (tuple(large), tuple(small))

                        total_men[i, j][pair].append(utility_random[i, j])
                        count_men[i, j][pair] += 1

            for j in self.women[1:]:
                for i in self.men:
                    if (i, j) not in self.leakage_arc:
                        large, small = [], []
                        for l in self.men:
                            if (l, j) not in self.leakage_arc and l != i:
                                if utility_random[j, l] > utility_random[j, i]:
                                    large.append(l)
                                else:
                                    small.append(l)

                                pair = (tuple(large), tuple(small))

                        total_women[j, i][pair].append(utility_random[j, i])
                        count_women[j, i][pair] += 1

        st = time.time()
        cond_men = {(i,j):defaultdict(float) for i in self.men[1:] for j in self.women if (i, j) not in self.leakage_arc}
        cond_women = {(j,i):defaultdict(float) for i in self.men for j in self.women[1:] if (i, j) not in self.leakage_arc}
        prob_men = {(i,j):defaultdict(float) for i in self.men[1:] for j in self.women if (i, j) not in self.leakage_arc}
        prob_women = {(j,i):defaultdict(float) for i in self.men for j in self.women[1:] if (i, j) not in self.leakage_arc}

        for i in self.men[1:]:
            for j in self.women:
                trun_M = min(M, len(count_men[i, j]))
                tops = sorted(count_men[i, j].items(), key=lambda item: item[1], reverse=True)[:trun_M]
                #tops_sum = sum([freq for pair, freq in tops])
                for pair, freq in tops:
                    prob_men[i, j][pair] = freq / Y
                    cond_men[i, j][pair] = np.mean(total_men[i, j][pair])

        for j in self.women[1:]:
            for i in self.men:
                trun_M = min(M, len(count_women[j, i]))
                tops = sorted(count_women[j, i].items(), key=lambda item: item[1], reverse=True)[:trun_M]
                #tops_sum = sum([freq for pair, freq in tops])
                for pair, freq in tops:
                    prob_women[j, i][pair] = freq / Y
                    cond_women[j, i][pair] = np.mean(total_women[j, i][pair])

        et = time.time()

        return cond_men, cond_women, prob_men, prob_women
