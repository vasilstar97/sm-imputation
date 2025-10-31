import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from .base import BaseImputer
import geopandas as gpd

GEOM_COLS = ['centroid_x', 'centroid_y']

class SknnImputer(BaseImputer):

    def __init__(self, *args, n_neighbors : int = 5, algorithm : str = 'ball_tree', **kwargs):
        self.n_neighbors = n_neighbors
        self.algorithm = algorithm
        super().__init__(*args, **kwargs)

    def _preprocess_blocks_gdf(self, blocks_gdf : gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        blocks_gdf = blocks_gdf.copy()
        gdf = super()._preprocess_blocks_gdf(blocks_gdf)
        gdf[GEOM_COLS[0]] = blocks_gdf.centroid.x
        gdf[GEOM_COLS[1]] = blocks_gdf.centroid.y
        return gdf

    def _get_neighbors(self, known_gdf : gpd.GeoDataFrame, unknown_gdf : gpd.GeoDataFrame) -> dict[int, gpd.GeoDataFrame]:
        nbrs = NearestNeighbors(n_neighbors=self.n_neighbors, algorithm=self.algorithm)
        nbrs.fit(known_gdf[GEOM_COLS])
        
        distances, indices = nbrs.kneighbors(unknown_gdf[GEOM_COLS])
        index = unknown_gdf.index.to_list()
        return {index[i]: (
            known_gdf.iloc[idx_list],
            distances[i]
        ) for i,idx_list in enumerate(indices)}

    def _impute(self, known_gdf : gpd.GeoDataFrame, unknown_gdf : gpd.GeoDataFrame):
        features_cols = self.features_cols
        
        neighbors = self._get_neighbors(known_gdf, unknown_gdf)

        imputed_values = []
        for _, d in neighbors.items():
            gdf, _ = d
            imputed_values.append(gdf[features_cols].mean(axis=0))

        return pd.DataFrame(imputed_values, columns=features_cols, index=unknown_gdf.index)