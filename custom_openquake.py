from openquake.hmtk.parsers.catalogue import *
from openquake.hmtk.plotting.seismicity.catalogue_plots import *
from openquake.hmtk.seismicity.declusterer.dec_gardner_knopoff import *
from openquake.hmtk.seismicity.declusterer.distance_time_windows import *
from openquake.hmtk.seismicity.selector import *
from openquake.hazardlib.geo.polygon import Polygon as PolyOQ
from openquake.hazardlib.geo.point import Point as PointOQ
from openquake.hazardlib.geo.line import Line as LineOQ
from openquake.hmtk.seismicity.completeness.comp_stepp_1971 import Stepp1971
from openquake.hmtk.seismicity.occurrence.b_maximum_likelihood import BMaxLikelihood
from openquake.hmtk.plotting.seismicity.completeness.plot_stepp_1972 import create_stepp_plot
from openquake.hmtk.seismicity.smoothing.smoothed_seismicity import SmoothedSeismicity
from openquake.hmtk.seismicity.smoothing.kernels.isotropic_gaussian import IsotropicGaussian
from openquake.hazardlib.geo.surface.complex_fault import ComplexFaultSurface
from shapely.geometry import *
from shapely.ops import *
import os
import utm
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import pickle

# membuat colormap diskrit
def get_cmap(n, name='jet'):
    return plt.cm.get_cmap(name, n)

def catalogue_to_dict(catalogue):
    idcs = ['eventID','year','month','day','hour',
       'minute','second','longitude','latitude',
       'depth','magnitude','sigmaMagnitude']

    dict_catalogue = dict()
    for idx in idcs:
        if isinstance(catalogue[idx], list): 
            dict_catalogue[idx] = catalogue[idx]
        else:
            dict_catalogue[idx] = catalogue[idx].tolist()
    return dict_catalogue
            
# save variabel ke pickle file untuk dibuka kembali di file lain
def variable_to_pkl(variabel, filename):
    with open(filename, 'wb') as file:
        pickle.dump(variabel, file, pickle.HIGHEST_PROTOCOL)            

# save katalog ke pickle file
def catalogue_to_pkl(catalogue, filename):
    dict_catalogue = catalogue_to_dict(catalogue)
    variable_to_pkl(dict_catalogue, filename)

# membuka pickle file
def open_pkl(filename):
    with open(filename, 'rb') as file:
        variabel = pickle.load(file)
    return variabel

# fungsi untuk mengubah data lon lat pada patahan menjadi poligon
def polygon_from_fault(fault_longitude, fault_latitude, distance=20):
    distance = distance * 1000
    fault1_utm = utm.from_latlon(np.array(fault_latitude), np.array(fault_longitude))
    input_for_faultLine1 = np.array([fault1_utm[0], fault1_utm[1]]).T.tolist()
    faultLine1 = LineString(input_for_faultLine1)
    dilated = faultLine1.buffer(distance)
    dilated_x, dilated_y = dilated.exterior.xy
    res_lat, res_lon = utm.to_latlon(np.array(dilated_x.tolist()), np.array(dilated_y.tolist()), fault1_utm[2], fault1_utm[3])
    return res_lon, res_lat

# polygon coords to shapely polygon
def poly_to_shapelypoly(poly):
    polygon = Polygon(np.array([poly[0], poly[1]]).T)
    return polygon

def shapely_to_OQ_poly(shapely_polygon):
    x, y = shapely_polygon.exterior.xy
    polyOQ1 = PolyOQ([PointOQ(lon, lat) for lon, lat in zip(x, y)])
    return polyOQ1

def copy_cutPoly_cutDepth(catalogue, polygon, lower_depth = None, upper_depth = None):
    selector1 = CatalogueSelector(catalogue, create_copy = True)
    catArea = selector1.within_polygon(polygon)
    selector2 = CatalogueSelector(catArea)
    catAreaDepth = selector2.within_depth_range(lower_depth = lower_depth, upper_depth = upper_depth)
    return catAreaDepth

# membuat area dengan jarak tertentu dari patahan
def create_area_faults(faults, distance=20):
    area_faults = {"merged": None, "individual":None}
    area_faults_coords = {"merged": None, "individual":None}
    if faults["merged"] != None:
        polygon_faults = [
            [
                polygon_from_fault(fault[0], fault[1], distance=distance) for fault in selected
            ] for selected in faults["merged"]
        ]
        shapelypolygon_faults = [
            [
                poly_to_shapelypoly(poly) for poly in selected
            ] for selected in polygon_faults
        ]
        merged_shapely = [unary_union(selected) for selected in shapelypolygon_faults]
        area_faults["merged"] = [shapely_to_OQ_poly(selected) for selected in merged_shapely]
        area_faults_coords["merged"] = [selected.coords for selected in area_faults["merged"]]
    if faults["individual"] != None:
        polygon_faults = [
            polygon_from_fault(fault[0], fault[1], distance=distance) for fault in faults["individual"]
        ]
        shapelypolygon_faults = [poly_to_shapelypoly(poly) for poly in polygon_faults]
        area_faults["individual"] = [shapely_to_OQ_poly(poly) for poly in shapelypolygon_faults]
        area_faults_coords["individual"] = [selected.coords for selected in area_faults["individual"]]
    return area_faults, area_faults_coords

# membuat katalog dari area fault
def create_catalogue_from_area_faults(catalogue, area_faults, depth=20):
    catalogue_area_faults = {"depth": depth, "merged": None, "individual": None}

    if area_faults["merged"] != None:
        catalogue_area_faults["merged"] = [
            copy_cutPoly_cutDepth
            (
                catalogue, selected_area, catalogue_area_faults["depth"]
            ) for selected_area in area_faults["merged"]
        ]

    if area_faults["individual"] != None:
        catalogue_area_faults["individual"] = [
            copy_cutPoly_cutDepth
            (
                catalogue, selected_area, catalogue_area_faults["depth"]
            ) for selected_area in area_faults["individual"]
        ]
    return catalogue_area_faults

def catalogue_declustering(catalogue, declust_config):
    declustering = GardnerKnopoffType1()
    cluster_index, cluster_flag = declustering.decluster(catalogue, declust_config)

    selectorCatalogue = CatalogueSelector(catalogue, create_copy = True)
    catalogue_declustered = selectorCatalogue.select_catalogue(cluster_flag == 0)
    return catalogue_declustered

def create_catalogue_megathrusts(catalogue, megathrust_geoms, distance = 20, mesh_spacing=10.):
    catalogue_megathrusts = []
    for geom in megathrust_geoms:
        geom["upper"].sort(reverse=True, key=lambda x: x[0])
        geom["lower"].sort(reverse=True, key=lambda x: x[0])
        edges = [
            LineOQ([PointOQ(*coord) for coord in geom["upper"]]),
            LineOQ([PointOQ(*coord) for coord in geom["lower"]])
        ]
        megathrustOQ = ComplexFaultSurface.from_fault_data(edges, mesh_spacing=mesh_spacing)
        selector2 = CatalogueSelector(catalogue, create_copy = True)
        catalogue_megathrust = selector2.within_rupture_distance(megathrustOQ, distance=distance)
        catalogue_megathrusts.append(catalogue_megathrust)
    return catalogue_megathrusts

def create_catalogue_from_shallow_backgrounds(catalogue, shallow_background_geoms, lower_depth=20):
    poly_shallow_background = [
        PolyOQ([PointOQ(lon, lat) for lon, lat in zip(*geom)]) for geom in shallow_background_geoms
    ]
    
    catalogue_shallow_backgrounds = [
        copy_cutPoly_cutDepth(catalogue, poly, lower_depth) for poly in poly_shallow_background
    ]
    return catalogue_shallow_backgrounds

def magnitude_of_completeness(catalogue, comp_config, mag_range=[3., 5.]):
    completeness_algorithm = Stepp1971()
    old = completeness_algorithm.completeness(catalogue, comp_config)
    completeness_algorithm.simplify(mag_range=[3., 5.])
    completeness_table = completeness_algorithm.completeness_table
    return completeness_table, old

def b_a_value(catalogue, mle_config, completeness_table):
    recurrence = BMaxLikelihood()
    b_val, sigma_b, a_val, sigma_a = recurrence.calculate(catalogue, 
                                                         mle_config, 
                                                         completeness = completeness_table)
    return b_val, sigma_b, a_val, sigma_a

def plot_magnitude_time_density_with_Mc(catalogue, completeness_table, magnitude_bin_width, time_bin_width, cmap_dist='Pastel1', cmap_Mc='Dark2'):
    plt.set_cmap(cmap_dist)
    plot_magnitude_time_density(catalogue, magnitude_bin_width, time_bin_width)
    cmap = get_cmap(len(completeness_table[:,0]), cmap_Mc)
    for i, (v, h) in enumerate(zip(completeness_table[:,0],completeness_table[:,1])):
        plt.axvline(v, c=cmap(i), label=f"{v:.0f}, Mc {h:.2f}")
        plt.axhline(h, c=cmap(i))
    plt.legend()

def plot_observed_reccurence_with_Mc(catalogue, completeness_table, completeness_table_old, magnitude_bin_width, cmap_Mc='Dark2'):
    plot_observed_recurrence(catalogue, completeness_table_old, magnitude_bin_width)
    cmap = get_cmap(len(completeness_table[:,0]), cmap_Mc)
    for i, (v, h) in enumerate(zip(completeness_table[:,0],completeness_table[:,1])):
        plt.axvline(h, c=cmap(i), label=f"{v:.0f}, Mc {h:.2f}")
    plt.legend()
    
def remove_events_in_A_from_B(catalogue_A, catalogue_B):
    selector1 = CatalogueSelector(catalogue_A)
    flags = ~np.in1d(catalogue_A.data['eventID'], catalogue_B.data['eventID'])
    unique, counts = np.unique(flags, return_counts=True)
    double_counting = dict(zip(unique, counts)).get(False, 0)
    print("jumlah gempa yang dihapus: ", double_counting)
    catalogue_C = selector1.select_catalogue(flags)
    return catalogue_C