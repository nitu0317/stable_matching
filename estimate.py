import numpy as np
from gurobipy import *
from collections import *
from itertools import chain, combinations
import cvxpy as cp
from opt import *
from scenario import *
import time

class ESTIMATE():
    def __init__(self, men, women, leakage_arc, eta, utility):
        self.men = men
        self.women = women
        self.eta = eta
        self.utility = utility
        self.leakage_arc = leakage_arc
        self.match_opt = MATCH_OPT(men, women, leakage_arc, eta, utility)
        self.scenario = SCENARIO(men, women, leakage_arc, eta, utility)

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
                persist_value[key] += np.round(value / Y, 3)

        end_time = time.time()
        print("Time of SAA: " + str(end_time - start_time))

        return persist_value

    def GRE(self):
        persist_value = defaultdict(float)
        for i in self.men:
            for j in self.women:
                persist_value[i, j] = self.utility[i, j]

        return persist_value

    def ENT(self):
        persist_value = defaultdict(float)
        res_x, res_obj = self.match_opt.RAM()
        for i in range(len(self.men)):
            for j in range(len(self.women)):
                persist_value[self.men[i], self.women[j]] = res_x[i, j] * (len(self.men) + len(self.women))

        return persist_value

    def SLP(self, seed, Y, M):
        persist_value = defaultdict(float)

        cond_men, cond_women, prob_men, prob_women = self.scenario.preference_pair(seed, M, Y)
        model, x, x_men, x_women = self.match_opt.matching_conditional_pair(cond_men, cond_women, prob_men, prob_women)
        res_x = model.getAttr('X', x)
        #res_x_men = model.getAttr('X', x_men)
        #res_x_women = model.getAttr('X', x_women)
        for key, value in res_x.items():
            persist_value[key] = np.round(value, 5)

        return persist_value
