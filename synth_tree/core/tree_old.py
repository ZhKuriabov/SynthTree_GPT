from .node import Node
from .pruning import Pruner
from .linear_model import LocalLinearModel
import numpy as np
from itertools import combinations
from tqdm import tqdm

class SynthTree:
    def __init__(self, task='regression', max_depth=None, min_samples_split=10, pruning_method=None, pruning_max_depth=3):
        self.task = task
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None
        self.pruning_method = pruning_method
        self.pruning_max_depth = pruning_max_depth

    def fit(self, location_matrix, model_matrix):
        self.root = self._grow_tree(location_matrix, model_matrix, depth=0)
        if self.pruning_method:
            pruner = Pruner(method=self.pruning_method, max_depth=self.pruning_max_depth)
            self.root = pruner.prune(self.root)

    def _pairwise_distance(self, M):
        dists = []
        for i, j in combinations(range(M.shape[0]), 2):
            if self.task == 'regression':
                d = np.mean((M[i] - M[j]) ** 2)
            else:
                a = (M[i] == 1) & (M[j] == 1)
                b = (M[i] == 1) & (M[j] == 0)
                c = (M[i] == 0) & (M[j] == 1)
                tp = np.sum(a)
                fp = np.sum(b)
                fn = np.sum(c)
                d = (fp + fn) / (2 * tp + 1e-6)
            dists.append(d)
        return np.mean(dists) if dists else 0

    def _goodness_of_split(self, M_parent, M_left, M_right):
        n = M_parent.shape[0]
        n_left = M_left.shape[0]
        n_right = M_right.shape[0]
        D_parent = self._pairwise_distance(M_parent)
        D_left = self._pairwise_distance(M_left)
        D_right = self._pairwise_distance(M_right)
        return D_parent - (n_left / n) * D_left - (n_right / n) * D_right

    def _grow_tree(self, X, M, depth):
        node = Node()
        if depth == self.max_depth or len(X) < self.min_samples_split or M.shape[0] <= 1:
            node.make_leaf(X, M, task=self.task)
            return node

        best_gain = -np.inf
        best_split = None

        for feature_index in tqdm(range(X.shape[1]), desc=f"Depth {depth}", leave=False):
            thresholds = np.unique(X[:, feature_index])
            for threshold in thresholds:
                left_mask = X[:, feature_index] <= threshold
                right_mask = ~left_mask
                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue
                M_left = M[left_mask]
                M_right = M[right_mask]
                gain = self._goodness_of_split(M, M_left, M_right)
                if gain > best_gain:
                    best_gain = gain
                    best_split = (feature_index, threshold, left_mask, right_mask)

        if best_split is None or best_gain <= 0:
            node.make_leaf(X, M, task=self.task)
            return node

        feature_index, threshold, left_mask, right_mask = best_split
        node.split_feature = feature_index
        node.split_value = threshold
        node.left = self._grow_tree(X[left_mask], M[left_mask], depth + 1)
        node.right = self._grow_tree(X[right_mask], M[right_mask], depth + 1)
        return node

    def predict(self, X):
        return np.array([self.root.predict(x) for x in X])

    def visualize(self, feature_names=None):
        def _print(node, depth):
            indent = "  " * depth
            if node.is_leaf:
                print(f"{indent}Leaf: model with prediction = {node.prediction}")
            else:
                fname = feature_names[node.split_feature] if feature_names else f"x{node.split_feature}"
                print(f"{indent}Split: {fname} <= {node.split_value:.3f}")
                _print(node.left, depth + 1)
                _print(node.right, depth + 1)
        _print(self.root, 0)
