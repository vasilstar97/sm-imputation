import pandas as pd
import geopandas as gpd
import osmnx as ox
import pyproj
import shapely
from blocksnet.enums import LandUse

BC_TAGS = {
    'roads': {
      "highway": ["construction","crossing","living_street","motorway","motorway_link","motorway_junction","pedestrian","primary","primary_link","raceway","residential","road","secondary","secondary_link","services","tertiary","tertiary_link","track","trunk","trunk_link","turning_circle","turning_loop","unclassified",],
      "service": ["living_street", "emergency_access"]
    },
    'railways': {
      "railway": "rail"
    },
    'water': {
      'riverbank':True,
      'reservoir':True,
      'basin':True,
      'dock':True,
      'canal':True,
      'pond':True,
      'natural':['water','bay'],
      'waterway':['river','canal','ditch'],
      'landuse':'basin',
      'water': 'lake'
    }
}

LU_RULES = {
    'residential': LandUse.RESIDENTIAL,
    'education': LandUse.RESIDENTIAL,

    'commercial': LandUse.BUSINESS,
    'retail': LandUse.BUSINESS,
    'religious': LandUse.BUSINESS,
    
    'forest': LandUse.RECREATION,
    'recreation_ground': LandUse.RECREATION,
    'grass': LandUse.RECREATION,
    'greenery': LandUse.RECREATION,
    
    'industrial': LandUse.INDUSTRIAL,
    'quarry': LandUse.INDUSTRIAL,

    'military': LandUse.SPECIAL,
    'cemetery': LandUse.SPECIAL,
    'landfill': LandUse.SPECIAL,
    
    'farmland': LandUse.AGRICULTURE,
    'animal_keeping': LandUse.AGRICULTURE,
    'greenhouse_horticulture': LandUse.AGRICULTURE,
    'plant_nursery': LandUse.AGRICULTURE,
    'vineyard': LandUse.AGRICULTURE,
    'allotments': LandUse.AGRICULTURE,

    'highway': LandUse.TRANSPORT,
    'railway': LandUse.TRANSPORT,
    'depot': LandUse.TRANSPORT
}

IS_LIVING_TAGS = ['residential', 'house', 'apartments', 'detached', 'terrace', 'dormitory']

def _get_urban_objects(boundaries_geom : shapely.geometry.base.BaseGeometry) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    water_gdf = ox.features_from_polygon(boundaries_geom, BC_TAGS['water'])
    roads_gdf = ox.features_from_polygon(boundaries_geom, BC_TAGS['roads'])
    railways_gdf = ox.features_from_polygon(boundaries_geom, BC_TAGS['railways'])

    water_gdf = water_gdf[water_gdf.geom_type.isin(['Polygon', 'MultiPolygon', 'LineString', 'MultiLineString'])]
    roads_gdf = roads_gdf[roads_gdf.geom_type.isin(['LineString', 'MultiLineString'])]
    railways_gdf = railways_gdf[railways_gdf.geom_type.isin(['LineString', 'MultiLineString'])]

    return (gdf.reset_index(drop=True) for gdf in [water_gdf, roads_gdf, railways_gdf])

def _get_land_use(boundaries_geom : shapely.geometry.base.BaseGeometry) -> gpd.GeoDataFrame:
    functional_zones_gdf = ox.features_from_polygon(boundaries_geom, tags={'landuse': True})
    return functional_zones_gdf[functional_zones_gdf.geom_type.isin(['Polygon', 'MultiPolygon'])].reset_index()

def _get_buildings(boundaries_geom : shapely.geometry.base.BaseGeometry, crs) -> gpd.GeoDataFrame:
    
    from blocksnet.preprocessing.imputing import impute_buildings
    
    buildings_gdf = ox.features_from_polygon(boundaries_geom, tags={'building': True})
    buildings_gdf = buildings_gdf.reset_index(drop=True).to_crs(crs)
    
    buildings_gdf['is_living'] = buildings_gdf['building'].apply(lambda b : b in IS_LIVING_TAGS)
    buildings_gdf['number_of_floors'] = pd.to_numeric(buildings_gdf['building:levels'], errors='coerce')

    return impute_buildings(buildings_gdf)

def _aggregate_buildings(blocks_gdf, buildings_gdf) -> gpd.GeoDataFrame:

    from blocksnet.blocks.aggregation import aggregate_objects

    agg_df, _ = aggregate_objects(blocks_gdf, buildings_gdf)

    return blocks_gdf.join(agg_df[['build_floor_area', 'footprint_area']])


def _assign_land_use(blocks_gdf, functional_zones_gdf) -> gpd.GeoDataFrame:
    
    from blocksnet.blocks.assignment import assign_land_use

    functional_zones_gdf = functional_zones_gdf.rename(columns={'landuse':'functional_zone'})
    blocks_gdf = assign_land_use(blocks_gdf, functional_zones_gdf, LU_RULES)
    return blocks_gdf


def _get_blocks_gdf(boundaries_gdf, boundaries_geom) -> gpd.GeoDataFrame:

    from blocksnet.blocks.cutting import preprocess_urban_objects, cut_urban_blocks
    from blocksnet.blocks.postprocessing import postprocess_urban_blocks

    water_gdf, roads_gdf, railways_gdf = _get_urban_objects(boundaries_geom)

    for gdf in [water_gdf, roads_gdf, railways_gdf]:
        gdf.to_crs(boundaries_gdf.crs, inplace=True)

    lines_gdf, polygons_gdf = preprocess_urban_objects(roads_gdf, railways_gdf, water_gdf)
    blocks_gdf = cut_urban_blocks(boundaries_gdf, lines_gdf, polygons_gdf)

    return postprocess_urban_blocks(blocks_gdf)

def _get_boundaries_gdf(city_name : str) -> gpd.GeoDataFrame:
    boundaries_gdf = ox.geocode_to_gdf(city_name)
    return boundaries_gdf.reset_index(drop=True)

def _estimate_crs(boundaries_gdf : gpd.GeoDataFrame) -> tuple[pyproj.CRS, shapely.geometry.base.BaseGeometry]:
    boundaries_geom = boundaries_gdf.union_all()
    crs = boundaries_gdf.estimate_utm_crs()
    return crs, boundaries_geom

def get_blocks_gdf(city_name : str) -> gpd.GeoDataFrame:
    boundaries_gdf = _get_boundaries_gdf(city_name)
    local_crs, boundaries_geom = _estimate_crs(boundaries_gdf)
    boundaries_gdf = boundaries_gdf.to_crs(local_crs)
    blocks_gdf = _get_blocks_gdf(boundaries_gdf, boundaries_geom)

    functional_zones_gdf = _get_land_use(boundaries_geom).to_crs(local_crs)
    blocks_gdf = _assign_land_use(blocks_gdf, functional_zones_gdf)

    buildings_gdf = _get_buildings(boundaries_geom, local_crs)
    blocks_gdf = _aggregate_buildings(blocks_gdf, buildings_gdf)

    return blocks_gdf

