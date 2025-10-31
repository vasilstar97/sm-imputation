import geopandas as gpd
import pandas as pd
from sklearn.decomposition import NMF
from .sknn import SknnImputer, GEOM_COLS

SITE_AREA_COLUMN = 'site_area'

class SmvNmfImputer(SknnImputer):

    def __init__(
        self, 
        *args, 
        additional_cols : list[str],
        n_components : int = 10,
        max_iter: int = 200,
        random_state: int = 42, 
        **kwargs
    ):
        self.additional_cols = additional_cols
        self.n_components = n_components
        self.max_iter = max_iter
        self.random_state = random_state
        super().__init__(*args, **kwargs)

    def _preprocess_blocks_gdf(self, blocks_gdf):
        blocks_gdf = blocks_gdf.copy()
        gdf = super()._preprocess_blocks_gdf(blocks_gdf)
        extra_columns = [*self.additional_cols, SITE_AREA_COLUMN]
        gdf[extra_columns] = blocks_gdf[extra_columns]
        return gdf
    
    def _impute(self, known_gdf : gpd.GeoDataFrame, unknown_gdf : gpd.GeoDataFrame):
        
        features_cols = self.features_cols
        additional_cols = self.additional_cols
        
        unknown_gdf.loc[unknown_gdf.index, features_cols] = super()._impute(known_gdf, unknown_gdf)

        all_gdf = pd.concat([known_gdf, unknown_gdf])
        x_smooth = all_gdf[[*GEOM_COLS, SITE_AREA_COLUMN, *additional_cols, *features_cols]].to_numpy()

        model = NMF(n_components=self.n_components, init='random',
                    max_iter=self.max_iter, random_state=self.random_state)
        W = model.fit_transform(x_smooth)
        H = model.components_
        x_imputed = W @ H

        imputed_values = x_imputed[-len(unknown_gdf):, -len(features_cols):]

        return pd.DataFrame(imputed_values, columns=features_cols, index=unknown_gdf.index)