import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression, Lasso
from sklearn.metrics import mean_squared_error, log_loss
from sklearn.base import BaseEstimator

class ConstantClassifier(BaseEstimator):
    def __init__(self, value):
        self.value = value

    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.full(X.shape[0], self.value)

class LocalLinearModel:
    def __init__(self, task='regression', alpha=0.01):
        self.task = task
        self.alpha = alpha
        self.model = None
        self.X = None
        self.y = None

    def fit(self, X, M):
        self.X = X
        self.y = M[:, 0] if M.ndim > 1 else M
        if self.task == 'classification':
            if len(np.unique(self.y)) == 1:
                self.model = ConstantClassifier(self.y[0])
            else:
                self.model = LogisticRegression()
                self.model.fit(X, self.y)
        else:
            self.model = Lasso(alpha=self.alpha, fit_intercept=True, max_iter=1000)
            self.model.fit(X, self.y)

    def predict(self, X, proba=False):
        if self.task == 'classification' and proba and hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)[:, 1]
        return self.model.predict(X)

    def loss(self):
        y_pred = self.predict(self.X)
        if self.task == 'regression':
            return mean_squared_error(self.y, y_pred)
        else:
            if len(np.unique(self.y)) == 1:
                return 0.0
            return log_loss(self.y, np.clip(y_pred, 1e-9, 1 - 1e-9))
