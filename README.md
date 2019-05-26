# BayesDNN
Implementation of Bayesian Deep Neural Network outlined from articles "Yarin Gal and Zoubin Ghahramani. Dropout as a bayesian approximation: Representing model uncer-tainty in deep learning" and "Mattias Teye, Hossein Azizpour, Kevin Smith and Bayesian Uncertainty Estimation for Batch Normalized Deep Networks".

## Getting Started

### Installing
```
conda env create -f BDNN_env.yml. 
```

The current version have tensorflow-gpu, so reinstall plain tensorflow if no gpu is available.

## Feedforward_example

This folder contains a Bayes Feedforward layer. Only synthetic data is used here. The figure below is one exmaple both batch normalization and droput is used to approximate a Gaussian processs.

![alt text](/Feedforward_example/assets/BN_DO.jpg)

## RNN_example

Here a time series forecast of antibiotica resistance is fitted with a Bayesian RNN. The figure below demonstrates a forecast for Pencillin resistance

![alt text](/RNN_example/temp/fig/Pencillin.png)

## Authors

* **Markus Ekvall**
