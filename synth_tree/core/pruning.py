import numpy as np

class Pruner:
    def __init__(self, method='cost_complexity', max_depth=3):
        self.method = method
        self.max_depth = max_depth

    def prune(self, tree, X=None, y=None):
        if self.method == 'cost_complexity':
            return self._cost_complexity_pruning(tree)
        elif self.method == 'depth':
            return self._depth_pruning(tree, self.max_depth)
        return tree

    def _resubstitution_error(self, node):
        if node.is_leaf:
            return node.model.loss() if node.model else 0.0
        else:
            return self._resubstitution_error(node.left) + self._resubstitution_error(node.right)

    def _number_of_leaf_nodes(self, node):
        if node.is_leaf:
            return 1
        return self._number_of_leaf_nodes(node.left) + self._number_of_leaf_nodes(node.right)

    def _subtrees(self, node):
        if node.is_leaf:
            return []
        return [(node, self._resubstitution_error(node), self._number_of_leaf_nodes(node))] + \
               self._subtrees(node.left) + self._subtrees(node.right)

    def _collect_data(self, node):
        if node.is_leaf:
            return node.X, node.model.y.reshape(-1, 1)
        X_left, M_left = self._collect_data(node.left)
        X_right, M_right = self._collect_data(node.right)
        X_combined = np.vstack([X_left, X_right])
        M_combined = np.vstack([M_left, M_right])
        return X_combined, M_combined

    def _cost_complexity_pruning(self, root):
        while True:
            subtrees = self._subtrees(root)
            if not subtrees:
                break
            if self._number_of_leaf_nodes(root) <= 2:
                break
            # Compute alpha = (R(t) - R(T_t)) / (|T_t| - 1)
            alphas = []
            for node, R_Tt, T_t_size in subtrees:
                if T_t_size <= 1:
                    continue
                R_t = node.model.loss() if node.is_leaf and node.model else 0
                alpha = (R_t - R_Tt) / (T_t_size - 1)
                alphas.append((alpha, node))
            if not alphas:
                break
            _, prune_node = min(alphas, key=lambda x: x[0])
            X_sub, M_sub = self._collect_data(prune_node)
            prune_node.left = None
            prune_node.right = None
            prune_node.make_leaf(X_sub, M_sub, task='classification')
        return root

    def _depth_pruning(self, node, max_depth=3, current_depth=0):
        if node is None:
            return None
        if current_depth >= max_depth:
            X_sub, M_sub = self._collect_data(node)
            node.left = None
            node.right = None
            node.make_leaf(X_sub, M_sub, task='classification')
            return node
        if node.left:
            node.left = self._depth_pruning(node.left, max_depth, current_depth + 1)
        if node.right:
            node.right = self._depth_pruning(node.right, max_depth, current_depth + 1)
        return node