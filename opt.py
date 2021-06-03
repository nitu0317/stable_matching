import numpy as np
from gurobipy import *
from collections import *
from itertools import chain, combinations
import cvxpy as cp

class MATCH_OPT():
    def __init__(self, men, women, leakage_arc, eta, utility):
        self.men = men
        self.women = women
        self.eta = eta
        self.utility = utility
        self.leakage_arc = leakage_arc

    def matching(self, stable=True, realization=None):
        model = Model()

        model.setParam("LogToConsole", 0)
        model.setParam("Presolve", 0)

        if realization is None:
            x = model.addVars([(i, j) for i in self.men for j in self.women if (i, j) not in self.leakage_arc], vtype=GRB.CONTINUOUS, lb=0, ub=1, name='match')
        else:
            x = model.addVars([(i, j) for i in self.men for j in self.women if (i, j) not in self.leakage_arc], vtype=GRB.BINARY, name='match')

        if realization is None:
            model.setObjective(sum([x[i, j] * (self.utility[i, j] + self.utility[j, i]) for i, j in x]), GRB.MAXIMIZE)
        else:
            model.setObjective(sum([x[i, j] * (realization[i, j] + realization[j, i]) for i in self.men[1:] for j in self.women[1:] if (i, j) not in self.leakage_arc]), GRB.MAXIMIZE)

        ## assignself.ment constraints
        for i in self.men[1:]:
            model.addConstr(sum([x[i, j] for j in self.women if (i, j) not in self.leakage_arc]) == 1)

        for j in self.women[1:]:
            model.addConstr(sum([x[i, j] for i in self.men if (i, j) not in self.leakage_arc]) == 1)

        ## stable constraints
        if stable:
            for i in self.men[1:]:
                for j in self.women[1:]:
                    if (i, j) not in self.leakage_arc:
                        lhs = x[i, j] + sum([x[i, k] for k in self.women if realization[i, j] > realization[i, k] and (i, k) not in self.leakage_arc]) + sum([x[l, j] for l in self.men if realization[j, i] > realization[j, l] and (l, j) not in self.leakage_arc])
                        rhs = 1
                        model.addConstr(lhs <= rhs)

#             ## truncation
#             j = self.women[0]
#             for i in self.men[1:]:
#                 model.addConstr(sum(x[i, k] for k in self.women if realization[i, j] > realization[i, k] and (i, k) not in self.leakage_arc) == 0)

#             i = self.men[0]
#             for j in self.women[1:]:
#                 model.addConstr(sum(x[l, j] for l in self.men if realization[j, i] > realization[j, l] and (l, j) not in self.leakage_arc) == 0)

        model.optimize()

        return model, x

    def matching_conditional(self, cond_men, cond_women, prob_men, prob_women):
        model = Model('persist_lhs')
        model.setParam("LogToConsole", 0)

        x = model.addVars([(i, j) for i in self.men for j in self.women if (i, j) not in self.leakage_arc], vtype=GRB.CONTINUOUS, lb=0, ub=1, name='match')

        x_men = {}
        for (i, j), rank_dict in cond_men.items():
            for rank in rank_dict:
                x_men[i, j, rank] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=1, name="self.men[%s,%s,%s]"%(i, j, rank[:10]))

        x_women = {}
        for (j, i), rank_dict in cond_women.items():
            for rank in rank_dict:
                x_women[i, j, rank] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=1, name="self.women[%s,%s,%s]"%(i, j, rank[:10]))


        reward_men = 0
        for i in self.men[1:]:
            for j in self.women[1:]:
                if (i, j) not in self.leakage_arc:
                    for rank, cond_u in cond_men[i, j].items():
                        reward_men += cond_u * x_men[i, j, rank] * prob_men[i][rank]

        reward_women = 0
        for j in self.women[1:]:
            for i in self.men[1:]:
                if (i, j) not in self.leakage_arc:
                    for rank, cond_u in cond_women[j, i].items():
                        reward_women += cond_u * x_women[i, j, rank] * prob_women[j][rank]

        model.setObjective(reward_men + reward_women, GRB.MAXIMIZE)

        ## assignself.ment constraints
#         for i in self.men[1:]:
#             model.addConstr(sum([x[i, j] for j in self.women if (i, j) not in self.leakage_arc]) == 1)

#         for j in self.women[1:]:
#             model.addConstr(sum([x[i, j] for i in self.men if (i, j) not in self.leakage_arc]) == 1)

        for i in self.men[1:]:
            for rank in prob_men[i]:
                model.addConstr(sum([x_men[i, j, rank] for j in self.women if (i, j) not in self.leakage_arc]) == 1)

        for j in self.women[1:]:
            for rank in prob_women[j]:
                model.addConstr(sum([x_women[i, j, rank] for i in self.men if (i, j) not in self.leakage_arc]) == 1)

        ## stable constraints
        for i in self.men[1:]:
            for j in self.women[1:]:
                if (i, j) not in self.leakage_arc:
                    lhs_1 = x[i, j]
                    lhs_2 = 0
                    for k in self.women:
                        if (i, k) not in self.leakage_arc:
                            for rank in cond_men[i, k]:
                                if rank.index(j) < rank.index(k):
                                    lhs_2 += prob_men[i][rank] * x_men[i, k, rank]
                    lhs_3 = 0
                    for l in self.men:
                        if (l, j) not in self.leakage_arc:
                            for rank in cond_women[j, l]:
                                if rank.index(i) < rank.index(l):
                                    lhs_3 += prob_women[j][rank] * x_women[l, j, rank]
                    rhs = 1
                    model.addConstr(lhs_1 + lhs_2 + lhs_3 <= rhs)

        ## marginal to conditional
        for i in self.men[1:]:
            for j in self.women:
                if (i, j) not in self.leakage_arc:
                    lhs = x[i, j]
                    rhs = 0
                    for rank in cond_men[i, j]:
                        rhs += prob_men[i][rank] * x_men[i, j, rank]
                    model.addConstr(lhs == rhs)

        for j in self.women[1:]:
            for i in self.men:
                if (i, j) not in self.leakage_arc:
                    lhs = x[i, j]
                    rhs = 0
                    for rank in cond_women[j, i]:
                        rhs += prob_women[j][rank] * x_women[i, j, rank]
                    model.addConstr(lhs == rhs)

        model.optimize()

        return model, x, x_men, x_women

    def matching_conditional_pair(self, cond_men, cond_women, prob_men, prob_women):
        model = Model('persist_lhs')
        model.setParam("LogToConsole", 0)

        x = model.addVars([(i, j) for i in self.men for j in self.women if (i, j) not in self.leakage_arc], vtype=GRB.CONTINUOUS, lb=0, ub=1, name='match')

        x_men = {}
        for (i, j), pair_dict in cond_men.items():
            for pair in pair_dict:
                x_men[i, j, pair] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=1, name="self.men")

        x_women = {}
        for (j, i), pair_dict in cond_women.items():
            for pair in pair_dict:
                x_women[i, j, pair] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=1, name="self.women")


        reward_men = 0
        for i in self.men[1:]:
            for j in self.women:
                if (i, j) not in self.leakage_arc:
                    for pair, cond_u in cond_men[i, j].items():
                        reward_men += cond_u * x_men[i, j, pair] * prob_men[i, j][pair]

        reward_women = 0
        for j in self.women[1:]:
            for i in self.men:
                if (i, j) not in self.leakage_arc:
                    for pair, cond_u in cond_women[j, i].items():
                        reward_women += cond_u * x_women[i, j, pair] * prob_women[j, i][pair]

        model.setObjective(reward_men + reward_women, GRB.MAXIMIZE)

        ## assignself.ment constraints
        for i in self.men[1:]:
            model.addConstr(sum([x[i, j] for j in self.women if (i, j) not in self.leakage_arc]) == 1)

        for j in self.women[1:]:
            model.addConstr(sum([x[i, j] for i in self.men if (i, j) not in self.leakage_arc]) == 1)


        ## stable constraints
        for i in self.men[1:]:
            for j in self.women[1:]:
                if (i, j) not in self.leakage_arc:
                    lhs_1 = x[i, j]
                    lhs_2 = 0
                    for k in self.women:
                        if (i, k) not in self.leakage_arc and k != j:
                            for pair in cond_men[i, k]:
                                large, small = pair[0], pair[1]
                                if j in large:
                                    lhs_2 += prob_men[i, k][pair] * x_men[i, k, pair]

                    lhs_3 = 0
                    for l in self.men:
                        if (l, j) not in self.leakage_arc and l != i:
                            for pair in cond_women[j, l]:
                                large, small = pair[0], pair[1]
                                if i in large:
                                    lhs_3 += prob_women[j, l][pair] * x_women[l, j, pair]
                    rhs = 1
                    model.addConstr(lhs_1 + lhs_2 + lhs_3 <= rhs)

        ## marginal to conditional
        for i in self.men[1:]:
            for j in self.women:
                if (i, j) not in self.leakage_arc:
                    lhs = x[i, j]
                    rhs = 0
                    for pair in cond_men[i, j]:
                        rhs += prob_men[i, j][pair] * x_men[i, j, pair]
                    model.addConstr(lhs == rhs)

        for j in self.women[1:]:
            for i in self.men:
                if (i, j) not in self.leakage_arc:
                    lhs = x[i, j]
                    rhs = 0
                    for pair in cond_women[j, i]:
                        rhs += prob_women[j, i][pair] * x_women[i, j, pair]
                    model.addConstr(lhs == rhs)

        model.optimize()

        return model, x, x_men, x_women

    def RAM(self):
        n_men, n_women = 1 / (len(self.men) + len(self.women) - 2), 1 / (len(self.men) + len(self.women) - 2)
        X = cp.Variable((len(self.men), len(self.women)))
        A = np.ones(len(self.women)-1)
        B = np.ones(len(self.men)-1)
        R = np.zeros((len(self.men), len(self.women)))
        for i in range(len(self.men)):
            for j in range(len(self.women)):
                R[i, j] = (self.utility[self.men[i], self.women[j]] + self.utility[self.women[j], self.men[i]])

        # define and solve the CVXPY problem.
        entropy_men = 0
        for i in range(1, len(self.men)):
            entropy_men += (cp.sum(cp.entr(X[i, :] * (len(self.men) + len(self.women) - 2))) * n_men)

        entropy_women = 0
        for j in range(1, len(self.women)):
            entropy_women += (cp.sum(cp.entr(X[:, j] * (len(self.men) + len(self.women) - 2))) * n_women)

        objective = cp.trace(R[1:,1:].T @ X[1:,1:]) + (entropy_men + entropy_women) * self.eta

        model = cp.Problem(cp.Maximize(objective), [X[1:,1:] @ A <= n_men, X[1:,1:].T @ B <= n_women, X >= 0])
        model.solve()
        res_x = X.value
        res_obj = model.value

        return res_x, res_obj
