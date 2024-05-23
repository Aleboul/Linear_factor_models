# Description

## Overview

This project simulates a data generation process for a Linear Factor Model and estimates the model parameters using extreme value theory. The process involves generating synthetic data with specific dependencies, calculating block maxima, and applying empirical rank transformations to derive extreme value coefficients. These coefficients are then used to estimate the matrix A that represents the linear combination of factors influencing the data.

## File descriptions

**`est_impure.py`**

The provided code consists of a set of functions designed to estimate the matrix A in a Linear Factor Model based on given data.

**`est_pure.py`**

This file contains all the helping functions to estimate the set of pure variables. The whole methodology is described by Theorem 2 in the main paper.

**`model_2.py`**

This code contains all the necessary elements to reproduce experiments in Section 5.2 when we consider a fixed block size m and varying k and d.

**`model_3.py`**

This code contains all the necessary elements to reproduce experiments in Section 5.2 when we consider a fixed block size n and varying k, m and d.


## Usage

Ensures the scripts (`est_impure.py`, `est_pure.py`) are in the same directory or properly referenced. Install any required packages (e.g, `numpy`, `pandas`, `clayton`, `scipy`).

Ensures to create folders `results/results_model_2` or `results/results_model_3` before running the scripts.

Execute `model_2.py` or `model_3.py`.

```python
>>> python3 model_2.py
>>> python3 model_3py
```
The results will be saved in the `results/` directory. Each subdirectory within `results/` will correspond to a specific simulation run and contain the estimated and original parameter matrices

## Notes

The simulation parameters (e.g, number of variables, blocks, dimensionality, sample size, columns, and iterations) can be adjusted in `model_2.py` or `model_3.py` to explore different scenarios. The estimated matrices can be compared with original matrices to evaluate the performance of the estimation method.
