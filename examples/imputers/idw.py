import geopandas as gpd
import pandas as pd
import numpy as np
from .sknn import SknnImputer

class IdwImputer(SknnImputer):

    def _impute(self, known_gdf : gpd.GeoDataFrame, unknown_gdf : gpd.GeoDataFrame):
        features_cols = self.features_cols
        
        neighbors = self._get_neighbors(known_gdf, unknown_gdf)

        imputed_values = []
        for _, d in neighbors.items():
            neighbors_df, distance = d
            distance = np.clip(distance, 1e-6, None)
            weights = 1 / (distance ** 2)
            weights /= weights.sum()

            values = neighbors_df[features_cols].values
            weighted_mean = (values * weights[:, None]).sum(axis=0)
            imputed_values.append(weighted_mean)

        return pd.DataFrame(imputed_values, columns=features_cols, index=unknown_gdf.index)