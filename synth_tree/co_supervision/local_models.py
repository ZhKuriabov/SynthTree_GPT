# synth_tree/co_supervision/local_models.py
from sklearn.linear_model import LinearRegression, LogisticRegression
import numpy as np

class LocalModelFitter:
    def __init__(self, task='classification'):
        self.task = task
        self.models = {}

    def fit_per_cluster(self, X, y, cluster_labels):
        for cluster_id in np.unique(cluster_labels):
            indices = np.where(cluster_labels == cluster_id)[0]
            X_cluster = X[indices]
            y_cluster = y[indices]
            model = LogisticRegression() if self.task == 'classification' else LinearRegression()
            model.fit(X_cluster, y_cluster)
            self.models[cluster_id] = model

    def predict_matrix(self, X):
        # Create prediction matrix M where each column is g_j(x_i) over all x_i
        M = []
        for model in self.models.values():
            preds = model.predict(X)
            M.append(preds)
        return np.column_stack(M)

    def predict(self, X, cluster_labels):
        y_pred = np.zeros(len(X))
        for cluster_id in np.unique(cluster_labels):
            indices = np.where(cluster_labels == cluster_id)[0]
            model = self.models[cluster_id]
            y_pred[indices] = model.predict(X[indices])
        return y_pred
