from abc import ABC, abstractmethod
import geopandas as gpd

FEATURES_COLS = ['footprint_area', 'build_floor_area']

class BaseImputer(ABC):

    def __init__(self, blocks_gdf : gpd.GeoDataFrame, features_cols : list[str] = FEATURES_COLS):
        self.features_cols = features_cols
        self.blocks_gdf = self._preprocess_blocks_gdf(blocks_gdf)

    def _preprocess_blocks_gdf(self, blocks_gdf : gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        features_cols = self.features_cols
        return blocks_gdf[features_cols].copy()

    def _split(self, blocks_ids : list[int]) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
        blocks_gdf = self.blocks_gdf
        known_gdf = blocks_gdf[~blocks_gdf.index.isin(blocks_ids)].copy()
        unknown_gdf = blocks_gdf[blocks_gdf.index.isin(blocks_ids)].copy()
        return known_gdf, unknown_gdf

    @abstractmethod
    def _impute(self, known_gdf : gpd.GeoDataFrame, unknown_gdf : gpd.GeoDataFrame):
        pass

    def impute(self, blocks_ids : list[int]):
        features_cols = self.features_cols
        known_gdf, unknown_gdf = self._split(blocks_ids)
        imputed_gdf = self._impute(known_gdf.copy(), unknown_gdf.copy())
        unknown_gdf.loc[blocks_ids, features_cols] = imputed_gdf.loc[blocks_ids, features_cols]
        return imputed_gdf
