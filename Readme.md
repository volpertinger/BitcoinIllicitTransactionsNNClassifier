# Dataset

[Elliptic Data Set](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set)

[1] Elliptic, www.elliptic.co.

[2] M. Weber, G. Domeniconi, J. Chen, D. K. I. Weidele, C. Bellei, T. Robinson, C. E. Leiserson, "Anti-Money Laundering
in Bitcoin: Experimenting with Graph Convolutional Networks for Financial Forensics", KDD ’19 Workshop on Anomaly
Detection in Finance, August 2019, Anchorage, AK, USA.

# Workflow

## Dataset preprocessing

Preparing raw elliptic dataset for further user in neural network:

Making split into 3 parts (inputs - x and outputs - y):

- Train part, using for NN training
- Test part, using for NN validation during training
- Validation part, for validation after training

Prettify data:

- Delete unnecessary columns
- Delete headers
- Rename classes

Save split data to .csv files

## Dataset Analysing

Analyse dataset internal:

- Plot bars
  - Show plots
  - Save plots
- Print dataset heads

## Model training

- Train model
- Save model

## Analysing train result

- Loading model if we need
- Plot metrics

## Model testing

- Loading model if we need
- Print validation metrics
- Loop with model predictions on validation dataset by indexes from terminal
