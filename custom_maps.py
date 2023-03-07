import os
import pickle
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, LineString, Polygon, MultiPolygon
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import glob
import shutil
import shapely

# folder penyimpanan
dir_figs = os.path.join(os.getcwd(), 'figs')
os.makedirs(dir_figs, exist_ok=True)

dir_json = os.path.join(os.getcwd(), 'json_output')
os.makedirs(dir_json, exist_ok=True)

def get_cmap(n, name='jet'):
    return plt.cm.get_cmap(name, n)

# save variabel ke pickle file untuk dibuka kembali di file lain
def variable_to_pkl(variabel, filename, protocol=False):
    with open(filename, 'wb') as file:
        if protocol==True:
            pickle.dump(variabel, file, pickle.HIGHEST_PROTOCOL)
        else:
            pickle.dump(variabel, file)
        
# membuka pickle file
def open_pkl(filename, recursive=False):
    if recursive == False:
        with open(filename, 'rb') as file:
            variable = pickle.load(file)
    else:
        list_variables = glob.glob(filename)
        variable = [open_pkl(var) for var in list_variables]
    return variable

def list_filenames(filename):
    list_variables = glob.glob(filename)
    return list_variables

def duplicate_var(var, multiplier):
    res = [var]*multiplier
    return res

def random_colors(fill = None, edge = None, name='jet', n = 1):
    color = []
    if (fill == None) and (edge == None):
        tmp = get_cmap(n*2, name)
        color = [[tmp(i), tmp(i+n)] for i in range(n)]
    elif fill == None:
        tmp = get_cmap(n, name)
        color = [[tmp(i), edge] for i in range(n)]
    elif edge == None:
        tmp = get_cmap(n, name)
        color = [[fill, tmp(i)] for i in range(n)]
    return color
        
# menyeleksi patahan merged atau individual
def select_faults(gdf, dict_config, dict_name="Segment__1"):
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
    dict_config["name"] = gdf[dict_name].tolist()
    return faults, dict_config

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

def check_megathrust_delimiter(gdf_megathrusts, idx=0, delimiter_number=4, ax=None, map_limit=None, skipped_indices = [1, None]):
    if ax == None:
        fig, ax = plt.subplots(figsize=(8,8))
    x, y = gdf_megathrusts.geometry[idx].exterior.coords.xy
    print(f"memilih {delimiter_number} dari {list(range(len(x)))}")
    k, l = skipped_indices
    ax.plot(x[k:delimiter_number], y[k:delimiter_number], c='b', label='b')
    ax.plot(x[delimiter_number:l], y[delimiter_number:l], c='r', label='r')

    ax.legend()
    if map_limit != None:
        ax.set_ylim(map_limit[0])
        ax.set_xlim(map_limit[1])
    plt.show()

def create_megathrust_coords(gdf_megathrusts, idcs, skipped_indices, upper_configs=None, upper_depth=0, lower_depth=50):
    megathrusts = []
    for i, (idx, upper_config, indices) in enumerate(zip(idcs, upper_configs, skipped_indices)):
        x, y = gdf_megathrusts.geometry[i].exterior.coords.xy
        k, l = indices
        megathrust = None
        if upper_config == 'r':
            megathrust = {
                "upper": [(xi, yi, upper_depth) for xi, yi in zip(x[idx:l], y[idx:l])],
                "lower": [(xi, yi, lower_depth) for xi, yi in zip(x[k:idx], y[k:idx])]
            }
        else:
            megathrust = {
                "lower": [(xi, yi, lower_depth) for xi, yi in zip(x[idx:l], y[idx:l])],
                "upper": [(xi, yi, upper_depth) for xi, yi in zip(x[k:idx], y[k:idx])]
            }
        megathrusts.append(megathrust)
    return megathrusts

def quick_create_map(list_shp_geometry=None, list_color_shp_geometry=None, list_label_shp_geometry=None,
                    list_catalogue=None, list_color_catalogue=None, list_label_catalogue=None, map_limit=None, ax=None, 
                     figsize=(8,8), markersize=5):
    if ax == None:
        fig, ax = plt.subplots(figsize=figsize)
    
    lines = []
    labels = []
    
    if list_shp_geometry != None:
        for filename, color, label in zip(list_shp_geometry, list_color_shp_geometry, list_label_shp_geometry):
            gdf = gpd.read_file(filename)
            gdf.plot(color=color[0], edgecolor=color[1], linewidth=1, ax=ax, label=label)
            lines += [Line2D([0], [0], linestyle="none", marker="s", markersize=10, 
               markeredgecolor=color[1], markerfacecolor=color[0])]
        labels += list_label_shp_geometry
    
    if list_catalogue != None:
        for filename, color, label in zip(list_catalogue, list_color_catalogue, list_label_catalogue):
            dict_catalogue = open_pkl(filename)
            ax.scatter(dict_catalogue['longitude'], 
               dict_catalogue['latitude'], 
               c=color[0], s=markersize, edgecolors=color[1], linewidths=0.5, 
               label=label)
            lines += [Line2D([0], [0], linestyle="none", marker="s", markersize=10, 
               markeredgecolor=color[1], markerfacecolor=color[0])]
        labels += list_label_catalogue

    ax.legend(lines, labels, ncols=3)
    if map_limit != None:
        ax.set_ylim(map_limit[0])
        ax.set_xlim(map_limit[1])
    plt.show()
    return fig, ax

def geoms_to_shp(geoms, filename=None, dict_faults=None, range_deep=None, type_of=None):
    if type_of == "area fault":
        if dict_faults["merged"] != None:
            for idcs, geom in zip(dict_faults["merged"], geoms["merged"]):
                poly = Polygon(geom)
#                 polygon_to_shp(poly, "area_fault_"+"_".join([dict_faults["name"][i] for i in idcs])+".shp")
                polygon_to_shp(poly, "area_fault_"+"_".join([dict_faults["name"][idcs[0]], "others"])+".shp")
        if dict_faults["individual"] != None:
            for idx, geom in zip(dict_faults["individual"], geoms["individual"]):
                poly = Polygon(geom)
                polygon_to_shp(poly, "area_fault_"+dict_faults["name"][idx]+".shp")
                
def redistribute_vertices(geom, distance):
    if geom.geom_type == 'LineString':
        num_vert = int(round(geom.length / distance))
        if num_vert == 0:
            num_vert = 1
        return LineString(
            [geom.interpolate(float(n) / num_vert, normalized=True)
             for n in range(num_vert + 1)])
    elif geom.geom_type == 'MultiLineString':
        parts = [redistribute_vertices(part, distance)
                 for part in geom]
        return type(geom)([p for p in parts if not p.is_empty])
    else:
        raise ValueError('unhandled geometry %s', (geom.geom_type,))
        
def save_faults_geojson(gdf, dict_faults, distance = None):
    if dict_faults["merged"] != None:
        for idcs in dict_faults["merged"]:
            gdf_new = gdf.iloc[idcs]
            if distance != None:
                gdf_new.geometry = gdf_new.geometry.apply(redistribute_vertices,distance=0.005)
            gdf_new.to_file(os.path.join(
                dir_json, 
                "json_fault_"+"_".join([dict_faults["name"][idcs[0]], "others"])+".geojson"
            ), driver="GeoJSON")
    if dict_faults["individual"] != None:
        for idx in dict_faults["individual"]:
            gdf.iloc[[idx]].to_file(os.path.join(
                dir_json, 
                "json_fault_"+dict_faults["name"][idx]+".geojson"
            ), driver="GeoJSON")
            
def reverse_geom(geom):
    def _reverse(x, y, z=None):
        if z:
            return x[::-1], y[::-1], z[::-1]
        return x[::-1], y[::-1]

    return shapely.ops.transform(_reverse, geom)

def plot_line(line, ax = None):
    x, y = line.coords.xy
    x = np.array(x)
    y = np.array(y)
    if ax == None:
        fig, ax = plt.subplots()
    ax.quiver(x[:-1], y[:-1], x[1:]-x[:-1], y[1:]-y[:-1], scale_units='xy', angles='xy', scale=1)
    xlim, ylim = (list(ax.get_xlim()) + [x.min(), x.max()], list(ax.get_ylim()) + [y.min(), y.max()])
    ax.set_xlim([min(xlim), max(xlim)])
    ax.set_ylim([min(ylim), max(ylim)])
    return ax

def save_area_fault_geojson(gdf_area_faults, model="Example"):
    gdf_geology_area_faults = gpd.GeoDataFrame({ "id": [0], "model": [model] }, 
                                               geometry=[MultiPolygon(gdf_area_faults.geometry.tolist())])
    gdf_geology_area_faults.to_file(os.path.join(
                                        dir_json, 
                                        "json_area_fault_KumeringNorth_others.geojson"
                                    ), driver="GeoJSON")