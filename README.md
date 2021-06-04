
# Reducing Recommendation Inequality via Stable Matching with Random Preferences

This repository is the official implementation of [Reducing Recommendation Inequality via Stable Matching with Random Preferences].


## Requirements

To install requirements:

Gurobi for linear programming

CVXPY for convex programming

## Generate recommendations together with performance metrics

```
python main.py -eta 0.5 -f 20 -j 20 -runs 100 -Y 1000 -M 32 -gamma 0.2 -m SAA
```

#### Several key parameters:

-eta: scale parameter

-f: number of freelancers

-j: number of jobs

-runs: number of problem instances

-Y: number of samples in each instance

-M: number of sparse scenarios

-gamma: star ratio

-m: recommendation method

## Results

We use one example to show the outputs with different recommendation methods.

#### Greedy
```
python main.py -eta 0.5 -f 30 -j 30 -runs 50 -Y 500 -M 32 -gamma 0.2 -m GRE
```

recommendation accuracy with top K: [0.0581 0.1056 0.1405 0.1663 0.1855 0.2    0.2855 0.3644 0.4343 0.5018]

recommendation coverage with top K: [0.1935 0.1935 0.1935 0.1935 0.1935 0.1935 0.7574 0.9174 0.9574 0.9639]

recommendation count distribution for top 5: min 0.0, max 27.48, median 0.0, mean 5.0, std 10.03.

#### Entropy
```
python main.py -eta 0.5 -f 30 -j 30 -runs 50 -Y 500 -M 32 -gamma 0.2 -m ENT
```

recommendation accuracy with top K: [0.1565 0.2686 0.3598 0.4349 0.5021 0.5581 0.6049 0.6483 0.684  0.7189]

recommendation coverage with top K: [0.6839 0.9103 0.9581 0.9677 0.9677 0.9677 0.9677 0.9677 0.9677 0.9677]

recommendation count distribution for top 5: min 2.72, max 7.58, median 5.0, mean 5.0, std 1.22.

#### Sparse Linear Relaxation
```
python main.py -eta 0.5 -f 30 -j 30 -runs 50 -Y 500 -M 32 -gamma 0.2 -m SLP
```

recommendation accuracy with top K: [0.2231 0.3677 0.4746 0.5536 0.6172 0.6691 0.7114 0.7495 0.7795 0.8049]

recommendation coverage with top K: [0.6845 0.9277 0.9639 0.9677 0.9677 0.9677 0.9677 0.9677 0.9677 0.9677]

recommendation count distribution for top 5: min 2.72, max 7.58, median 5.0, mean 5.0, std 1.19.

#### Sample Average Approximation
```
python main.py -eta 0.5 -f 30 -j 30 -runs 50 -Y 500 -M 32 -gamma 0.2 -m SAA
```

recommendation accuracy with top K: [0.2453 0.4031 0.5199 0.6086 0.6761 0.7295 0.7694 0.8024 0.8307 0.8556]

recommendation coverage with top K: [0.7342 0.949  0.9658 0.9677 0.9677 0.9677 0.9677 0.9677 0.9677 0.9677]

recommendation count distribution for top 5: min 2.8, max 7.44, median 5.0, mean 5.0, std 1.14.
