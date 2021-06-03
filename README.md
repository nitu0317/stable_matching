
# Reducing Recommendation Inequality via Stable Matching with Random Preferences

This repository is the official implementation of [Reducing Recommendation Inequality via Stable Matching with Random Preferences].


## Requirements

To install requirements:


## Training

To train the model(s) in the paper, run this command:

```train
python train.py --input-data <path_to_data> --alpha 10 --beta 20
```

>📋  Describe how to train the models, with example commands on how to train the models in your paper, including the full training procedure and appropriate hyperparameters.

## Evaluation

To evaluate my model on ImageNet, run:

```eval
python eval.py --model-file mymodel.pth --benchmark imagenet
```

## Results

Our model achieves the following performance on :

### [Image Classification on ImageNet]

| Model name         | Top 1 Accuracy  | Top 5 Accuracy |
| ------------------ |---------------- | -------------- |
| My awesome model   |     85%         |      95%       |
