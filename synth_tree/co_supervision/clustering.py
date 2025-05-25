import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

class ClusteringManager:
    def __init__(self, max_k=15, min_k=2, method='silhouette'):
        self.max_k = max_k
        self.min_k = min_k
        self.method = method

    def find_best_k(self, X):
        scores = []
        for k in range(self.min_k, self.max_k + 1):
            kmeans = KMeans(n_clusters=k, random_state=0)
            labels = kmeans.fit_predict(X)
            score = silhouette_score(X, labels)
            scores.append((k, score))
        best_k = max(scores, key=lambda x: x[1])[0]
        return best_k, scores

    def cluster(self, X, k):
        kmeans = KMeans(n_clusters=k, random_state=0)
        labels = kmeans.fit_predict(X)
        return labels, kmeans.cluster_centers_
