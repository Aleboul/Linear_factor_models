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

MaxPrecipFallFrance=pd.read_csv('../data/MaxPrecipFallFrance.csv')
print(MaxPrecipFallFrance)
Longitudes = pd.read_csv('../data/LongitudesMaxPrecipFallFrance.csv')
Latitudes = pd.read_csv('../data/LatitudesMaxPrecipFallFrance.csv')

print(Longitudes)
print(Latitudes)
fig, ax = plt.subplots()
d = 92
colors = ['#6fa8dc', '#e06666', '#93c47d', '#FFA500']

A_hat = np.array(pd.read_csv('Ahat.csv', index_col = 0))
for latent in range(A_hat.shape[1]):
    index = np.where(A_hat[:,latent] > 0)[0]
    fig, ax = plt.subplots()
    polys1 = gpd.GeoSeries(Polygon([(-6,52), (10,52), (10,41), (-6,41)]))
    df1 = gpd.GeoDataFrame({'geometry':polys1}).set_crs('epsg:4326')
    world = gpd.read_file('../data/world-administrative-boundaries.geojson')
    ext_fra = gpd.overlay(world, df1, how='intersection')
    fra = gpd.read_file("../data/departements.geojson")

    fra.to_crs(epsg=4326)
    fra.boundary.plot(ax=ax, linewidth=.5, color='black')
    ext_fra.plot(ax=ax, color = 'lightgrey', linewidth = 0.1)
    fra.plot(ax=ax, color='white')
    ext_fra.boundary.plot(ax = ax,color = 'black', linewidth = 0.1)
    for i in index :
        print(i)
        x,y = Longitudes.x[i], Latitudes.x[i]
        delta = 0.15*A_hat[i,latent]
        cu = Polygon([((x-delta), (y-delta)), ((x+delta), (y-delta)),
                                       ((x+delta), (y+delta)), ((x-delta), (y+delta))])
        xs, ys = cu.exterior.xy
        ax.fill(xs, ys, fc=colors[latent], ec='none', alpha = A_hat[i,latent])
        ax.plot(xs, ys, color=colors[latent], linewidth=0.75)

    ax.set_aspect(1.0)
    ax.grid(False)
    fig.savefig('results/plot_map/cluster_' + str(latent)+'.pdf')