from dataclasses import dataclass, field
from typing import Any, Optional

import math
import numpy as np


@dataclass
class TreeNode:
    feature_idx: int = field(default=-1)
    feature_val: Any = field(default=None)
    is_leaf_node: bool = field(default=False)
    pred_value: Any = field(default=None)
    majority_class: Any = field(default=None, init=False)
    left: 'TreeNode' = field(default=None, init=False)
    right: 'TreeNode' = field(default=None, init=False)


class BaseRegressionTree:
    def __init__(
            self,
            lmbda: float,
            gamma: float,
            min_samples: int = 2,
            max_depth: int = 5,
            threshold: float = 0.01
    ):
        self.lmbda = lmbda
        self.gamma = gamma
        self.min_samples = min_samples
        self.max_depth = max_depth
        self.threshold = threshold
        self.root = None

    def square_sum_error(self, y_true: np.ndarray, y_pred: np.ndarray):
        return np.sum(np.square(y_true - y_pred))

    # y* - y
    def first_order(self, y_true: np.ndarray, y_pred: np.ndarray):
        return y_pred - y_true

    def second_order(self, y_true: np.ndarray, y_pred: np.ndarray):
        return np.ones_like(y_true)

    def calculate_pred_value(self, g: np.ndarray, h: np.ndarray):
        return -g.sum() / (h.sum() + self.lmbda)

    # 越大越好
    def greedy_score(self, g: np.ndarray, h: np.ndarray):
        """
        :param g: 一阶导
        :param h: 二阶导
        :return: 当前划分的分数
        """
        numerator = np.square(np.sum(g))
        denominator = 2 * (np.sum(h) + self.lmbda)
        return numerator / denominator

    # 划分增益
    def greedy_gain_by_feature_val(
            self,
            X: np.ndarray,
            g: np.ndarray,
            h: np.ndarray,
            feature_idx: int,
            partition_feature_val: float
    ):
        # g = self.first_order(y_true, y_pred)
        # h = self.second_order(y_true, y_pred)
        score1 = self.greedy_score(g, h)

        idx_less_eq = np.where(X[:, feature_idx] <= partition_feature_val)
        idx_great = np.where(X[:, feature_idx] > partition_feature_val)

        score2 = self.greedy_score(g[idx_less_eq], h[idx_less_eq])
        score3 = self.greedy_score(g[idx_great], h[idx_great])
        return (score2 + score3) - score1

    def greedy_gain_by_feature(
            self,
            X: np.ndarray,
            g: np.ndarray,
            h: np.ndarray,
            feature_idx: int
    ):
        feature_val_list = list(set(X[:, feature_idx]))
        feature_val_list = sorted(feature_val_list)

        best_feature_val  = feature_val_list[0]
        best_feature_gain = -math.inf

        for feature_val in feature_val_list:
            gain = self.greedy_gain_by_feature_val(X, g, h, feature_idx, feature_val)
            if gain > best_feature_gain:
                best_feature_val = feature_val
                best_feature_gain = gain

        return best_feature_val, best_feature_gain


    def greedy_gain(
            self,
            X: np.ndarray,
            # y_true: np.ndarray,
            # y_pred: np.ndarray,
            g: np.ndarray,
            h: np.ndarray
    ):
        # g = self.first_order(y_true, y_pred)
        # h = self.second_order(y_true, y_pred)
        feature_idx_list = list(range(X.shape[1]))
        best_feature_idx = 0
        best_feature_val = None
        best_feature_gain = -math.inf

        for feature_idx in feature_idx_list:
            val, gain = self.greedy_gain_by_feature(X, g, h, feature_idx)
            if gain > best_feature_gain:
                best_feature_idx = feature_idx
                best_feature_val = val
                best_feature_gain = gain
        return best_feature_idx, best_feature_val, best_feature_gain

    def build(self, X: np.ndarray, g: np.ndarray, h: np.ndarray, depth: int):
        # g = self.first_order(y_true, y_pred)
        # h = self.second_order(y_true, y_pred)
        # 边界条件
        # 样本个数 类别数 损失 深度
        num_samples, _ = X.shape
        # criterion = self.square_sum_error(y_true, y_pred)
        # criterion < self.threshold or \
        if num_samples <= self.min_samples or \
            depth >= self.max_depth:
            pred_value = self.calculate_pred_value(g, h)
            return TreeNode(-1, -1, True, pred_value)

        # 寻找最优划分特征和划分特征值
        feature_idx, feature_val, feature_gain = self.greedy_gain(X, g, h)

        # gamma理解为最低结构收益
        if feature_gain < self.gamma:
            pred_value = self.calculate_pred_value(g, h)
            return TreeNode(-1, -1, True, pred_value)
        node = TreeNode(feature_idx, feature_val)

        idx_left = np.where(X[:, feature_idx] <= feature_val)
        idx_right = np.where(X[:, feature_idx] > feature_val)
        node.left = self.build(X[idx_left], g[idx_left], h[idx_left], depth + 1)
        node.right = self.build(X[idx_right], g[idx_right], h[idx_right], depth + 1)
        return node

    def fit(self, X: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray):
        g = self.first_order(y_true, y_pred)
        h = self.second_order(y_true, y_pred)
        self.root = self.build(X, g, h, 0)
        return self

    def predict_single(self, x: np.ndarray, node: 'TreeNode'):
        if node.is_leaf_node:
            return node.pred_value
        if x[node.feature_idx] <= node.feature_val:
            return self.predict_single(x, node.left)
        else:
            return self.predict_single(x, node.right)

    def predict(self, X: np.ndarray):
        pred = []
        for x in X:
            pred.append(self.predict_single(x, self.root))
        return np.array(pred)


class XGBoost:
    def __init__(
            self,
            learning_rate: float = 0.1,
            gamma: float = 0.01,
            lmbda: float = 0.01,
            min_samples: int = 2,
            max_depth: int = 10,
            max_trees: int = 10,
            tree_threshold: float = 0.001
    ):

        self.gamma = gamma
        self.lmbda = lmbda
        self.min_samples = min_samples
        self.max_depth = max_depth
        self.max_trees = max_trees
        self.tree_threshold = tree_threshold
        self.learning_rate = learning_rate
        self.trees = []

    def build(self, X: np.ndarray, y: np.ndarray):
        y_pred = np.zeros_like(y)
        for _ in range(self.max_trees):
            tree = BaseRegressionTree(self.lmbda, self.gamma, self.min_samples, self.max_depth, self.tree_threshold)
            tree.fit(X, y, y_pred)
            if tree.root.left is None and tree.root.right is None:
                break
            self.trees.append(tree)
            y_pred += self.learning_rate * tree.predict(X)


    def fit(self, X: np.ndarray, y: Optional[np.ndarray]):
        if y is None:
            y = X[:, -1]
            X = X[:, :-1]
        self.build(X, y)
        return self

    def predict_single(self, x: np.ndarray):
        y_pred = 0
        for tree in self.trees:
            y_pred += self.learning_rate * tree.predict_single(x)
        return y_pred

    def predict(self, X: np.ndarray):
        y_pred = []
        for x in X:
            y_pred.append(self.predict_single(x))
        return np.array(y_pred)



