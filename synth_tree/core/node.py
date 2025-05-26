import numpy as np
from .linear_model import LocalLinearModel

class Node:
    def __init__(self):
        self.is_leaf = False
        self.split_feature = None
        self.split_value = None
        self.left = None
        self.right = None
        self.model = None
        self.prediction = None

    def make_leaf(self, X, M, task='regression'):
        self.is_leaf = True
        self.X = X
        self.M = M
        self.model = LocalLinearModel(task=task)
        self.model.fit(X, M)
        self.prediction = np.mean(M) if task == 'regression' else np.bincount(M[:, 0].astype(int)).argmax()

    def predict(self, x, proba=False):
        if self.is_leaf:
            return self.model.predict(x.reshape(1, -1), proba=proba)[0]
        elif x[self.split_feature] <= self.split_value:
            return self.left.predict(x, proba=proba)
        else:
            return self.right.predict(x, proba=proba)

    def count_leaves(self):
        if self.is_leaf:
            return 1
        return self.left.count_leaves() + self.right.count_leaves()

    def assign_leaf_ids(self, counter):
        if self.is_leaf:
            self.leaf_id = counter[0]
            counter[0] += 1
        else:
            self.left.assign_leaf_ids(counter)
            self.right.assign_leaf_ids(counter)

    def get_leaf_id(self, x):
        if self.is_leaf:
            return self.leaf_id
        if x[self.split_feature] <= self.split_value:
            return self.left.get_leaf_id(x)
        else:
            return self.right.get_leaf_id(x)

