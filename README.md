
# Reducing Recommendation Inequality via Stable Matching with Random Preferences

This repository is the official implementation of [Reducing Recommendation Inequality via Stable Matching with Random Preferences].


## Requirements

To install requirements:

Gurobi for linear programming

CVXPY for convex programming

## Create recommendations together with performance metrics

```
python main.py -eta 0.5 -f 20 -j 20 -runs 100 -Y 1000 -M 32 -gamma 0.2 -m SAA
```

## Results

Our model achieves the following performance on :

### [Image Classification on ImageNet]

| Model name         | Top 1 Accuracy  | Top 5 Accuracy |
| ------------------ |---------------- | -------------- |
| My awesome model   |     85%         |      95%       |
