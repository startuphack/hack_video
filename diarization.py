from sklearn.cluster import AgglomerativeClustering, DBSCAN
from files import pickle_load
import numpy as np


if __name__ == '__main__':
    vectors = pickle_load('speakers.gz.mdl')

    clustering = AgglomerativeClustering(n_clusters = None,linkage = 'complete',distance_threshold=0.55, affinity = 'cos')
    # clustering = DBSCAN(min_samples=1, metric='cosine')
    results = clustering.fit_predict(vectors)
    print(results)
