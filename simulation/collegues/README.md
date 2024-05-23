# Description

## Overview

This project simulates a data generation process for a Linear Factor Model and estimates the model parameters using extreme value theory. Extreme directions are estimated using both DAMEX Algorithm Nicolas Goix, Anne Sabourin, Stephan Clémençon, "Sparse representation of multivariate extremes with applications to anomaly detection", Journal of Multivariate Analysis, Volume 161, (2017) and our proposed procedure. Also, we compare sKmeans designed for extremes of Anja Janßen, Phyllis Wan "k-means clustering of extremes", Electronic Journal of Statistics (2020) with our method.

## File descriptions

**`damex.py`**

It contains necessary function to perform DAMEX Algorithm.

**`est_impure.py`**

The provided code consists of a set of functions designed to estimate the matrix A in a Linear Factor Model based on given data.

**`est_pure.py`**

This file contains all the helping functions to estimate the set of pure variables. The whole methodology is described by Theorem 2 in the main paper.

**`model_1.py`**

This code produces numerical experiments to compare sKmeans and SCRAM for estimating normalised columns of a matrix A in a Linear Factor Model with scaling and pure variable conditions.

**`model_1.R`**

Necessary file to perform sKmeans as given in the paper of Anja Janßen, Phyllis Wan "k-means clustering of extremes", Electronic Journal of Statistics (2020).

**`model_2.R`**

This code produces numerical experiments to compare DAMEX and SCRAM to detect extreme directions in a Linear Factor Model with scaling and pure variable conditions.

**`utilies.py`**

Miscellaneous functions for numerical experiments.

## Usage

Ensures the scripts (`est_impure.py`, `est_pure.py`, `damex.py`, `utilities.py`) are in the same directory or properly referenced. Install any required packages (e.g, `numpy`, `pandas`, `clayton`, `scipy`).

Ensures to create folders `results/results_model_1` and `results/results_model_2` before running the scripts.

Execute `model_1.py` or `model_2.py`.

```python
>>> python3 model_1.py
>>> python3 model_2.py
```
The results will be saved in the `results/` directory. Each subdirectory within `results/` will correspond to a specific simulation run and contain the estimated and original parameter matrices

## Notes

The simulation parameters (e.g, number of variables, blocks, dimensionality, sample size, columns, and iterations) can be adjusted in `model_1.py` or `model_2.py` to explore different scenarios.
