import numpy as np
from gurobipy import *
from collections import *
from itertools import chain, combinations
import cvxpy as cp
from opt import *
import time

class ESTIMATE():
    def __init__(self, men, women, leakage_arc, eta, utility):
        self.men = men
        self.women = women
        self.eta = eta
        self.utility = utility
        self.leakage_arc = leakage_arc
        self.match_opt = MATCH_OPT(men, women, leakage_arc, eta, utility)

    def SAA(self, seed, Y):
        persist_value = defaultdict(float)
        start_time = time.time()
        np.random.seed(seed)
        for _ in range(Y):
            utility_random = {}
            for i in self.men:
                for j in self.women:
                    epsilon = np.random.gumbel(0, self.eta)
                    utility_random[i, j]  = self.utility[i, j] + epsilon

                    epsilon = np.random.gumbel(0, self.eta)
                    utility_random[j, i] = self.utility[j, i] + epsilon

            model, x = self.match_opt.matching(True, utility_random)
            res_x = model.getAttr('X', x)
            for key, value in res_x.items():
                persist_value[key] += value / Y

        res = {key : np.round(persist_value[key], 3) for key in persist_value}
        end_time = time.time()
        print("Time of SAA: " + str(end_time - start_time))

        return res
