"""
    This code segment loads data related to maximum precipitation fall in France and
    coordinates of specific locations. It then visualizes the spatial distribution of
    precipitation clusters using GeoPandas and Matplotlib. The code iterates through each
    cluster, retrieves its corresponding indices, and plots them on a map. Each cluster is
    represented by a polygon shape, with its color intensity determined by the weight of the
    corresponding factor. The resulting visualizations provide insights into the geographical
    distribution and intensity of precipitation clusters across France.
"""

import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from random import sample

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import seaborn as sns
import geopandas as gpd
from shapely.geometry import Polygon
from matplotlib.patches import Rectangle

from itertools import islice
from shapely.ops import unary_union
plt.style.use('qb-light.mplstyle')

# Load the dataset containing information about maximum precipitation fall in France
MaxPrecipFallFrance = pd.read_csv('../data/MaxPrecipFallFrance.csv')
print(MaxPrecipFallFrance)

# Load the longitude and latitude coordinates for the locations
Longitudes = pd.read_csv('../data/LongitudesMaxPrecipFallFrance.csv')
Latitudes = pd.read_csv('../data/LatitudesMaxPrecipFallFrance.csv')

print(Longitudes)
print(Latitudes)

# Create subplots for visualization
fig, ax = plt.subplots()

# Define the number of clusters
d = 92

# Define colors for different clusters
colors = ['#6fa8dc', '#e06666', '#93c47d', '#FFA500']

# Load the matrix containing factor weights obtained from the model
A_hat = np.array(pd.read_csv('results/Ahat.csv', index_col=0))

# Iterate over each cluster
for latent in range(A_hat.shape[1]):
    # Retrieve indices of locations belonging to the current cluster
    index = np.where(A_hat[:, latent] > 0)[0]

    # Create a subplot for each cluster
    fig, ax = plt.subplots()

    # Define a polygon covering the entire France area
    polys1 = gpd.GeoSeries(Polygon([(-6, 52), (10, 52), (10, 41), (-6, 41)]))
    df1 = gpd.GeoDataFrame({'geometry': polys1}).set_crs('epsg:4326')

    # Load administrative boundaries of France
    world = gpd.read_file('../data/world-administrative-boundaries.geojson')

    # Extract the intersection area between France and the polygon
    ext_fra = gpd.overlay(world, df1, how='intersection')

    # Load departmental boundaries of France
    fra = gpd.read_file("../data/departements.geojson")

    # Set the coordinate reference system (CRS) to EPSG:4326
    fra.to_crs(epsg=4326)

    # Plot departmental boundaries
    fra.boundary.plot(ax=ax, linewidth=0.5, color='lightgrey')

    # Plot the intersection area
    ext_fra.plot(ax=ax, color='lightgrey', linewidth=0.1)

    # Fill the departmental areas with white color
    fra.plot(ax=ax, color='white')

    # Plot the boundary of the intersection area
    ext_fra.boundary.plot(ax=ax, color='black', linewidth=0.1)

    # Plot each location within the current cluster
    for i in index:
        # Retrieve longitude and latitude of the location
        x, y = Longitudes.x[i], Latitudes.x[i]

        # Calculate the size of the polygon based on the weight of the factor
        delta = 0.15 * A_hat[i, latent]

        # Define the coordinates of the polygon
        cu = Polygon([((x - delta), (y - delta)), ((x + delta), (y - delta)),
                      ((x + delta), (y + delta)), ((x - delta), (y + delta))])

        # Extract the x and y coordinates of the polygon
        xs, ys = cu.exterior.xy

        # Fill the polygon with a specific color based on the factor weight
        ax.fill(xs, ys, fc=colors[latent], ec='none', alpha=A_hat[i, latent])

        # Plot the polygon outline
        ax.plot(xs, ys, color=colors[latent], linewidth=0.75)

    # Set the aspect ratio of the plot and remove the grid
    ax.set_aspect(1.0)
    ax.grid(False)

    # Save the plot as a PDF file
    fig.savefig('results/plot_map/cluster_' + str(latent) + '.pdf')
