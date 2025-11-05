import geopandas as gpd
import pandas as pd
from .base import BaseImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from catboost import CatBoostClassifier

CLUSTER_COLUMN = 'cluster'
SITE_AREA_COLUMN = 'site_area'

class Spacematrix():

    def __init__(self, n_clusters : int, random_state : int):
        self.scaler = StandardScaler()
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init='auto')

    def _scale(self, df : pd.DataFrame) -> pd.DataFrame:
        scaler = self.scaler
        try:
            values = scaler.transform(df)
        except:
            values = scaler.fit_transform(df)
        return pd.DataFrame(values, columns=df.columns, index=df.index)
    
    def _clusterize(self, df : pd.DataFrame) -> pd.DataFrame:
        kmeans = self.kmeans
        try:
            result = kmeans.predict(df)
        except:
            result = kmeans.fit_predict(df)
        return pd.DataFrame(result, columns=[CLUSTER_COLUMN], index=df.index)

    def _preprocess(self, blocks_df : pd.DataFrame) -> pd.DataFrame:
        return blocks_df[['fsi', 'gsi']].copy()

    def run(self, blocks_df : pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        
        blocks_df = self._preprocess(blocks_df)

        mask = (blocks_df > 0).all(axis=1)
        df = blocks_df[mask].copy()

        scaled_df = self._scale(df)
        clusterize_df = self._clusterize(scaled_df)

        blocks_df[CLUSTER_COLUMN] = -1
        blocks_df.loc[df.index, CLUSTER_COLUMN] = clusterize_df.loc[df.index,CLUSTER_COLUMN]

        clusters_df = blocks_df.groupby('cluster').agg('median')

        return blocks_df, clusters_df

class SmImputer(BaseImputer):

    def __init__(
        self, 
        *args, 
        additional_cols : list[str],
        n_clusters : int = 11,
        random_state: int = 42, 
        **kwargs
    ):
        self.additional_cols = additional_cols
        self.n_clusters = n_clusters
        self.random_state = random_state
        super().__init__(*args, **kwargs)

    def _preprocess_blocks_gdf(self, blocks_gdf):
        blocks_gdf = blocks_gdf.copy()
        gdf = super()._preprocess_blocks_gdf(blocks_gdf)
        extra_columns = [*self.additional_cols, SITE_AREA_COLUMN]
        gdf[extra_columns] = blocks_gdf[extra_columns]
        return gdf
    
    def _spacematrix(self, known_gdf) -> tuple[pd.DataFrame, pd.DataFrame]:
        sm = Spacematrix(self.n_clusters, self.random_state)
        return sm.run(known_gdf)
    
    def _impute(self, known_gdf : gpd.GeoDataFrame, unknown_gdf : gpd.GeoDataFrame):

        features_cols = self.features_cols
        additional_cols = self.additional_cols
        classifier_cols = [SITE_AREA_COLUMN, *additional_cols]
        
        sm_df, clusters_df = self._spacematrix(known_gdf[features_cols])
        classifier = CatBoostClassifier(verbose=False)
        classifier.fit(known_gdf[classifier_cols], sm_df[CLUSTER_COLUMN])
        
        probs = classifier.predict_proba(unknown_gdf[classifier_cols])
        labels = classifier.classes_

        clusters_df = clusters_df.loc[labels]
        fsi_values = clusters_df['fsi'].values
        gsi_values = clusters_df['gsi'].values

        fsi_weighted = (probs * fsi_values).sum(axis=1)
        gsi_weighted = (probs * gsi_values).sum(axis=1)

        unknown_gdf['fsi'] = fsi_weighted
        unknown_gdf['gsi'] = gsi_weighted

        return unknown_gdf[features_cols].copy()