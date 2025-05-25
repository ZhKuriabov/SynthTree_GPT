import numpy as np

def augment_cluster_data(X_cluster, model, num_aug=100, noise_std=0.05):
    """
    Generate synthetic samples around X_cluster and label using model.predict.
    """
    mean = np.mean(X_cluster, axis=0)
    cov = np.cov(X_cluster.T) + np.eye(X_cluster.shape[1]) * 1e-6
    samples = np.random.multivariate_normal(mean, noise_std * cov, size=num_aug)
    y_synthetic = model.predict(samples)
    return samples, y_synthetic
