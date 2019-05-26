# BayesDNN
Implementation of Bayesian Deep Neural Network outlined from articles "Yarin Gal and Zoubin Ghahramani. Dropout as a bayesian approximation: Representing model uncer-tainty in deep learning" and "Mattias Teye, Hossein Azizpour, Kevin Smith and Bayesian Uncertainty Estimation for Batch Normalized Deep Networks".

## Getting Started

### Installing
```
conda env create -f BDNN_env.yml. 
```

The current version has tensorflow-gpu. Re-install to plain tensorflow if there is no gpu is available.

## Feedforward_example

This folder contains a Bayesian Feedforward layer. Only synthetic data have been tested. The figure below is one example where both batch normalization and dropout is used to approximate a Gaussian process.

![alt text](/Feedforward_example/assets/BN_DO.jpg)

## RNN_example

Here a time series forecast of antibiotic resistance is fitted with a Bayesian RNN. The figure below demonstrates a forecast for Penicillin resistance.

![alt text](/RNN_example/temp/fig/Pencillin.png)

## Authors

* **Markus Ekvall**
