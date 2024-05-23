"""
    This Python code segment performs geographical mapping and visualization
    of clusters derived from wildfire intensity data. Initially, it reads the
    wildfire dataset from a CSV file and extracts relevant coordinate information
    such as longitude and latitude. The code then removes duplicate coordinate
    entries to ensure uniqueness. Afterward, it iterates through each latent
    cluster identified in the dataset and plots them on a geographical map
    using the GeoPandas library.
    
    For each cluster, it generates a polygon centered around the variable's
    coordinates, with the polygon's size determined by the cluster's strength
    as encoded in the matrix A_hat. The polygons are plotted on the map with
    varying transparency and colors corresponding to different clusters.
    Additionally, administrative boundaries and geographic features are
    overlaid on the map to provide context.

    Finally, the resulting map visuals are saved as PDF files for presentation.
    Overall, this code facilitates the spatial visualization of wildfire clusters,
    aiding in the interpretation and understanding of the spatial distribution of
    wildfire intensity across geographic regions.
"""

import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from random import sample
from pyproj import Transformer

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import seaborn as sns
import geopandas as gpd
from shapely.geometry import Polygon
from matplotlib.patches import Rectangle

from itertools import islice
from shapely.ops import unary_union
plt.style.use('qb-light.mplstyle')

# Create a transformer object for coordinate system transformation
transformer = Transformer.from_crs("EPSG:27572", "EPSG:4326")

# Read the wildfire dataset from a CSV file
wildfire = pd.read_csv('../data/wildfire.csv')
print(wildfire)

# Extract relevant columns containing pixel IDs and coordinates
df_coord = wildfire[['PIX', 'X', 'Y']]
print(df_coord)

# Remove duplicate coordinate entries to ensure uniqueness
df_coord = df_coord.drop_duplicates()
print(df_coord)

# Extract longitude and latitude values from the DataFrame
Longitudes = df_coord.X
Latitudes = df_coord.Y

print(Longitudes)
print(Latitudes)

# Initialize a color palette for different clusters
colors = ['#6fa8dc', '#e06666', '#93c47d']

# Initialize figure
fig, ax = plt.subplots()

# Parameters
d = 1143
colors = ['#6fa8dc', '#e06666', '#93c47d']
A_hat = np.array(pd.read_csv('results/Ahat.csv', index_col=0))

# Iterate over each latent cluster identified in the dataset
for latent in range(A_hat.shape[1]):
    index = np.where(A_hat[:, latent] > 0)[0]

    # Plot geographical features on a map
    fig, ax = plt.subplots()
    polys1 = gpd.GeoSeries(Polygon(
        [(500000, 1600000), (1200000, 1600000), (1200000, 2100000), (500000, 2100000)]))
    df1 = gpd.GeoDataFrame({'geometry': polys1}).set_crs('epsg:27572')
    fra = gpd.read_file("../data/departements.geojson")
    fra.set_crs(epsg=4326)
    fra = fra.to_crs("epsg:27572")
    world = gpd.read_file('../data/world-administrative-boundaries.geojson')
    world = world.to_crs("epsg:27572")
    ext_fra = gpd.overlay(world, df1, how='intersection')
    int_fra = gpd.overlay(fra, df1, how="intersection")
    ext_fra = gpd.overlay(ext_fra, int_fra, how='difference')
    ext_fra.plot(ax=ax, color='lightgrey', linewidth=0.1)
    int_fra.boundary.plot(ax=ax, linewidth=.25, color='black')
    int_fra.plot(ax=ax, color='white')
    ext_fra.boundary.plot(ax=ax, color='black', linewidth=0.1)

    # Plot polygons representing clusters on the map
    for i in index:
        x, y = Longitudes.values[i]*10e2, Latitudes.values[i]*10e2
        delta = 50e2*A_hat[i, latent]
        cu = Polygon([((x-delta), (y-delta)), ((x+delta), (y-delta)),
                      ((x+delta), (y+delta)), ((x-delta), (y+delta))])
        xs, ys = cu.exterior.xy
        ax.fill(xs, ys, fc=colors[latent], ec='none', alpha=A_hat[i, latent])
        ax.plot(xs, ys, color=colors[latent], linewidth=0.75)

    # Set plot properties
    ax.set_aspect(1.0)
    ax.grid(False)

    # Save the plot as a PDF file
    fig.savefig('results/plot_map/cluster_' + str(latent)+'.pdf')
