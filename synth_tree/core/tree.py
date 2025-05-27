from .node import Node
from .pruning import Pruner
from .linear_model import LocalLinearModel
import numpy as np
from itertools import combinations
from tqdm import tqdm
from joblib import Parallel, delayed
import graphviz as gp

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

    def _evaluate_split(self, X, M, feature_index):
        thresholds = np.unique(X[:, feature_index])
        best_gain = -np.inf
        best_split = None
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
        return best_gain, best_split

    def _grow_tree(self, X, M, depth):
        node = Node()
        if depth == self.max_depth or len(X) < self.min_samples_split or M.shape[0] <= 1:
            node.make_leaf(X, M, task=self.task)
            return node

        print(f"Evaluating splits at depth {depth}...")
        results = Parallel(n_jobs=-1)(
            delayed(self._evaluate_split)(X, M, feature_index)
            for feature_index in range(X.shape[1])
        )
        print(f"Depth {depth}: {len(results)} features evaluated.")

        best_gain, best_split = max(results, key=lambda x: x[0] if x[1] is not None else -np.inf)

        if best_split is None or best_gain <= 0:
            node.make_leaf(X, M, task=self.task)
            return node

        feature_index, threshold, left_mask, right_mask = best_split
        node.split_feature = feature_index
        node.split_value = threshold
        node.left = self._grow_tree(X[left_mask], M[left_mask], depth + 1)
        node.right = self._grow_tree(X[right_mask], M[right_mask], depth + 1)
        return node

    def predict(self, X, proba=False):
        return np.array([self.root.predict(x, proba=proba) for x in X])
        
    def visualize_graphviz(self, feature_names=None, leaf_sizes_test=None):
    
        dot = gp.Digraph(graph_attr={"bgcolor": "white"})  # белый фон — опционально
    
        # определить максимальный размер (для нормализации цвета)
        max_size = max(leaf_sizes_test.values()) if leaf_sizes_test else 1
    
        def _add_nodes_edges(node, parent=None, label=""):
            if node.is_leaf:
                name = f"Leaf_{node.leaf_id}"
                base_label = f"Leaf {node.leaf_id}"
                size = leaf_sizes_test.get(node.leaf_id, 0) if leaf_sizes_test else 0
                green = int(255 * size / max_size) if max_size > 0 else 0
                fill = f"#{0:02x}{green:02x}{0:02x}"
                label = f"{base_label}\nsize={size}"
                dot.node(name, label=label, shape="box", style="filled", fillcolor=fill)
            else:
                name = f"Node_{id(node)}"
                fname = feature_names[node.split_feature] if feature_names else f"x{node.split_feature}"
                label = f"{fname} <= {node.split_value:.2f}"
                dot.node(name, label=label)
                _add_nodes_edges(node.left, name, "True")
                _add_nodes_edges(node.right, name, "False")
    
            if parent:
                dot.edge(parent, name, label=label)
    
        _add_nodes_edges(self.root)
        return dot
    
    def visualize(self, feature_names=None):
        def _print(node, depth, prefix=""):
            indent = "    " * depth
            if node.is_leaf:
                # Показываем топ-5 признаков
                if hasattr(node.model.model, 'coef_'):
                    coefs = np.abs(node.model.model.coef_)
                    top_indices = np.argsort(coefs)[::-1][:5]
                    top_features = [feature_names[i] if feature_names else f"x{i}" for i in top_indices]
                    print(f"{indent}{prefix}Leaf: top features = {top_features}")
                else:
                    value = getattr(node.model.model, 'value', 'undefined')
                    print(f"{indent}{prefix}Leaf: constant output = {value}")
            else:
                fname = feature_names[node.split_feature] if feature_names else f"x{node.split_feature}"
                print(f"{indent}{prefix}Split: {fname} <= {node.split_value:.3f}")
                _print(node.left, depth + 1, prefix="├─ Left: ")
                _print(node.right, depth + 1, prefix="└─ Right:")

        _print(self.root, 0)

    def print_leaf_models(self, feature_names=None):
        def _recurse(node, path):
            if node.is_leaf:
                print(" → ".join(path) if path else "Root")
                if hasattr(node.model.model, 'coef_'):
                    coefs = node.model.model.coef_
                    indices = np.argsort(np.abs(coefs))[::-1]
                    for i in indices[:10]:
                        name = feature_names[i] if feature_names else f"x{i}"
                        print(f"  {name:20s}: {coefs[i]:.4f}")
                elif hasattr(node.model.model, 'value'):
                    print(f"  Constant output: {node.model.model.value}")
                print()
            else:
                fname = feature_names[node.split_feature] if feature_names else f"x{node.split_feature}"
                _recurse(node.left, path + [f"{fname} <= {node.split_value:.2f}"])
                _recurse(node.right, path + [f"{fname} > {node.split_value:.2f}"])

        _recurse(self.root, [])

    def average_fraction_per_leaf(self, total_features):
        leaves = []

        def collect_leaves(node):
            if node.is_leaf:
                leaves.append(node)
            else:
                collect_leaves(node.left)
                collect_leaves(node.right)

        collect_leaves(self.root)

        fractions = []
        for leaf in leaves:
            if hasattr(leaf.model.model, 'coef_'):
                coef = leaf.model.model.coef_.flatten()
                num_nonzero = np.count_nonzero(coef)
                fractions.append(num_nonzero / total_features)

        return np.mean(fractions)

    def interpretability_score(self, X, total_features, alpha=0.5):
        # Weighted average sparsity
        leaf_counts = {}
        leaf_feature_counts = {}

        for x in X:
            node = self.root
            while not node.is_leaf:
                if x[node.split_feature] <= node.split_value:
                    node = node.left
                else:
                    node = node.right
            leaf_id = id(node)
            if leaf_id not in leaf_counts:
                leaf_counts[leaf_id] = 0
                if hasattr(node.model.model, 'coef_'):
                    coef = node.model.model.coef_.flatten()
                    leaf_feature_counts[leaf_id] = np.count_nonzero(coef)
                else:
                    leaf_feature_counts[leaf_id] = 0
            leaf_counts[leaf_id] += 1

        total_points = len(X)
        weighted_fraction = sum(
            (leaf_counts[lid] / total_points) * (leaf_feature_counts[lid] / total_features)
            for lid in leaf_counts
        )

        # Average path length
        def path_len(x):
            node = self.root
            depth = 0
            while not node.is_leaf:
                if x[node.split_feature] <= node.split_value:
                    node = node.left
                else:
                    node = node.right
                depth += 1
            return depth

        avg_path = np.mean([path_len(x) for x in X])

        return alpha * weighted_fraction + (1 - alpha) * avg_path

    def get_models_info(self):
        models = []
    
        def traverse(node):
            if node.is_leaf:
                # Попробуем извлечь коэффициенты и смещение
                if hasattr(node.model, "model") and hasattr(node.model.model, "coef_"):
                    coef = node.model.model.coef_.flatten()
                    intercept = node.model.model.intercept_
                else:
                    coef = np.zeros(node.X.shape[1])
                    intercept = 0.0
    
                models.append({
                    "coef": coef,
                    "intercept": intercept,
                    "n_samples": len(node.X),
                    "feature_names": node.model.feature_names if hasattr(node.model, "feature_names") else [f"x{i}" for i in range(coef.shape[0])]
                })
            else:
                traverse(node.left)
                traverse(node.right)
    
        traverse(self.root)
        return models

    def assign_leaf_ids(self):
        self.root.assign_leaf_ids(counter=[0])
    
    def get_leaf_id(self, x):
        return self.root.get_leaf_id(x)
