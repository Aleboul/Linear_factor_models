# Precipitation Analysis using Linear Factor Model

## Overview

This project focuses on analyzing precipitation maxima data for France using Linear Factor Models. The goal is to understand precipitation maxima patterns across different regions of France by identifying clusters and their relationship. The analysis involves employing Linear Factor Models to uncover underlying structures in the data and then visualizing the probabilities of exceeding precipitation thresholds for different clusters.

## File descriptions

**`est_impure.py`**

The provided code consists of a set of functions designed to estimate the matrix A in a Linear Factor Model based on given data.

**`est_pure.py`**

This file contains all the helping functions to estimate the set of pure variables. The whole methodology is described by Theorem 2 in the main paper.

**`lfm.py`**

This code performs the data-driven estimation of the matrix A in a Linear Factor Model.

**`map.py`**

This Python code segment performs geographical mapping and visualization of clusters derived from wildfire intensity data.

**`failure_set.py`**

This segment alculates and visualizes the probabilities of exceeding thresholds for different clusters identified by the Linear Factor Model.

## Usage

Ensures the scripts (`est_impure.py`, `est_pure.py`) are in the same directory or properly referenced. Install any required packages (e.g, `numpy`, `pandas`, `clayton`, `scipy`, `geopandas`).

Execute `lfm.py` then `map.py`.

```python
>>> python3 lfm.py
>>> python3 map.py
```
The first script will read the wildfire data, preprocess it, and apply SCRAM to identify spatial clusters. After clustering, the script will generate visualizations and save them in the results/plot_map directory.

## Data

The Data folder contains several csv files:

- `MaxPrecipFallFrance.csv`: Weekly maxima over 92 stations in France
- `LongitudesMaxPrecipFallFrance.csv`: Longitude of the 92 stations
- `LatitudesMaxPrecipFallFrance.csv`: Latitude of the 92 stations

## Results

The results of the precipitation analysis using Linear Factor Model will be stored in the `results` directory. This includes:

- Cluster plots (`cluster_*.pdf`) showing the spatial distribution of identified clusters.
- Failure set (`pmin.pdf`) showing the probability that at least one station (in a cluster) is above a certain threshold.
- Estimated matrix A (`Ahat.csv`) containing the cluster information.
- Associations with respect to Euclidean distance (`weights_distance_*.pdf`).