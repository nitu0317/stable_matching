import numpy as np
from gurobipy import *
from collections import *
from itertools import chain, combinations

class GRAPH():
    def customer_setup(seed, N_men, N_women, regime, gamma):
        men = ['mNA'] + ['m' + str(i) for i in range(N_men)]
        women = ['wNA'] + ['w' + str(j) for j in range(N_women)]

        np.random.seed(seed)
        utility = defaultdict(float)

        for i in men:
            for j in women:
                if i == 'mNA' or j == 'wNA':
                    utility[i, j] = 0
                    utility[j, i] = 0
                else:
                    utility[i, j] = np.round(np.random.uniform(regime[0], regime[1]), 2)
                    utility[j, i] = np.round(np.random.uniform(regime[0], regime[1]), 2)


        if gamma > 0:
            for j in women[1:]:
                for i in men[1:int(N_men * gamma)+1]:
                    utility[j, i] += 10

            for i in men[1:]:
                for j in women[1:int(N_women * gamma)+1]:
                    utility[i, j] += 10

        return men, women, utility
