from sklearn.cluster import AgglomerativeClustering, DBSCAN
from utils.files import pickle_load
import numpy as np

# https://wq2012.github.io/awesome-diarization/
# ! http://www.ifp.illinois.edu/~hning2/papers/Ning_spectral.pdf
# https://github.com/wq2012/SpectralCluster
if __name__ == '__main__':
    vectors = pickle_load('speakers.gz.mdl')

    clustering = AgglomerativeClustering(n_clusters = None,linkage = 'complete',distance_threshold=0.55, affinity = 'cos')
    # clustering = DBSCAN(min_samples=1, metric='cosine')
    results = clustering.fit_predict(vectors)
    print(results)
