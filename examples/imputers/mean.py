import geopandas as gpd
from .base import BaseImputer

class MeanImputer(BaseImputer):

    def _impute(self, known_gdf : gpd.GeoDataFrame, unknown_gdf : gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        features_cols = self.features_cols
        for feature_col in features_cols:
            mean_value = known_gdf[feature_col].mean()
            unknown_gdf[feature_col] = mean_value
        return unknown_gdf
        