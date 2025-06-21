from sklearn.cluster import KMeans
import pandas as pd

def run_clustering(df):
    model = KMeans(n_clusters=3)
    df['cluster'] = model.fit_predict(df)
    return df['cluster'].value_counts().to_dict()
