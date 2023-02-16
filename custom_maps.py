import os
import pickle
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, LineString, Polygon
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def get_cmap(n, name='jet'):
    return plt.cm.get_cmap(name, n)

# save variabel ke pickle file untuk dibuka kembali di file lain
def variable_to_pkl(variabel, filename):
    with open(filename, 'wb') as file:
        pickle.dump(variabel, file, pickle.HIGHEST_PROTOCOL)
        
# membuka pickle file
def open_pkl(filename):
    with open(filename, 'rb') as file:
        variabel = pickle.load(file)
    return variabel
        
# menyeleksi patahan merged atau individual
def select_faults(gdf, dict_config):
    faults = dict()
    faults["merged"] = None
    faults["individual"] = None
    if dict_config["merged"] != None:
        gdf_merged = [gdf.iloc[ls] for ls in dict_config["merged"]]
        faults["merged"] = [
        [
            [
                geom.coords.xy[0].tolist(), geom.coords.xy[1].tolist()
            ] for geom in selected.geometry]
            for selected in gdf_merged
        ]
    if dict_config["individual"] != None:
        gdf_ind = gdf.iloc[dict_config["individual"]]
        faults["individual"] = [
            [
                geom.coords.xy[0].tolist(), geom.coords.xy[1].tolist()
            ] for geom in gdf_ind.geometry
        ]
    return faults

def save_to_output_dir(gdf, name_file):
    savedir = os.path.join(os.getcwd(), 'shp_output', name_file.split('.')[0])
    os.makedirs(savedir, exist_ok=True)
    gdf.to_file(os.path.join(savedir, name_file))

def catalogue_to_shp(dict_catalogue, filename):
    df = pd.DataFrame.from_dict(dict_catalogue)
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude))
    save_to_output_dir(gdf, filename)

def polygon_to_shp(polygon, filename):
    gdf = gpd.GeoDataFrame(index=[0], crs='epsg:4326', geometry=[polygon])
    save_to_output_dir(gdf, filename)

def check_megathrust_delimiter(gdf_megathrusts, idx=0, delimiter_number=4, ax=None):
    if ax == None:
        fig, ax = plt.subplots(figsize=(8,8))
    x, y = gdf_megathrusts.geometry[idx].exterior.coords.xy
    print(f"memilih {delimiter_number} dari {list(range(len(x)))}")
    ax.plot(x[1:delimiter_number], y[1:delimiter_number], c='b', label='b')
    ax.plot(x[delimiter_number::], y[delimiter_number::], c='r', label='r')

    ax.legend()
    ax.set_ylim((-9, -2))
    ax.set_xlim((100, 110))
    plt.show()

def create_megathrust_coords(gdf_megathrusts, idcs, upper_configs=None, upper_depth=0, lower_depth=50):
    megathrusts = []
    for i, (idx, upper_config) in enumerate(zip(idcs, upper_configs)):
        x, y = gdf_megathrusts.geometry[i].exterior.coords.xy
        megathrust = None
        if upper_config == 'r':
            megathrust = {
                "upper": [(xi, yi, upper_depth) for xi, yi in zip(x[idx::], y[idx::])],
                "lower": [(xi, yi, lower_depth) for xi, yi in zip(x[1:idx], y[1:idx])]
            }
        else:
            megathrust = {
                "lower": [(xi, yi, lower_depth) for xi, yi in zip(x[idx::], y[idx::])],
                "upper": [(xi, yi, upper_depth) for xi, yi in zip(x[1:idx], y[1:idx])]
            }
        megathrusts.append(megathrust)
    return megathrusts