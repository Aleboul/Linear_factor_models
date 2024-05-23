# Wildfire Clustering Analysis

## Overview

This project aims to analyze wildfire data using clustering techniques to identify spatial patterns and clusters of wildfires. The dataset contains information about wildfire risks, including pixel IDs and coordinates. The main goal is to cluster the Forest Weather Index based on their spatial distribution and characteristics.

## File descriptions

**`est_impure.py`**

The provided code consists of a set of functions designed to estimate the matrix A in a Linear Factor Model based on given data.

**`est_pure.py`**

This file contains all the helping functions to estimate the set of pure variables. The whole methodology is described by Theorem 2 in the main paper.

**`lfm.py`**

This code performs the data-driven estimation of the matrix A in a Linear Factor Model.

**`map.py`**

This Python code segment performs geographical mapping and visualization of clusters derived from wildfire intensity data.

## Usage

Ensures the scripts (`est_impure.py`, `est_pure.py`) are in the same directory or properly referenced. Install any required packages (e.g, `numpy`, `pandas`, `clayton`, `scipy`, `geopandas`).

Execute `lfm.py` then `map.py`.

```python
>>> python3 lfm.py
>>> python3 map.py
```
The first script will read the wildfire data, preprocess it, and apply SCRAM to identify spatial clusters. After clustering, the script will generate visualizations and save them in the results/plot_map directory.

## Data

The wildfire dataset (`wildfire.csv`) contains the following columns:

- `PIX`: Pixel ID
- `X`: Longitude
- `Y`: Latitude
- `FWI`: Fire Weather Index (FWI)

## Results

The results of the wildfire clustering analysis will be stored in the `results` directory. This includes:

- Cluster plots (`cluster_*.pdf`) showing the spatial distribution of identified clusters.
- Estimated matrix A (`Ahat.csv`) containing the cluster information.