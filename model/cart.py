from collections import Counter
from dataclasses import dataclass
from typing import Any, List, Callable

import numpy as np
from math import inf
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


# 计算基尼系数
def gini_ratio(y: np.ndarray):
    # n,
    y = y.reshape((-1,)).tolist()
    n = len(y)
    cnt = Counter(y)
    p_sum = 0.
    for val in cnt:
        p = cnt[val] / n
        p_sum += p ** 2
    return 1 - p_sum


# 越小数据越集中
def mean_square_error(y: np.ndarray):
    c = y.mean()
    score = (y - c) ** 2
    return score.mean()

# 计算离散特征下，按照某个值划分数据集得到的指标
def discrete_feature_partition(
        X: np.ndarray,
        y: np.ndarray,
        feature_idx: int,
        feature_val: Any,
        score_func: Callable = mean_square_error
):
    n, _ = X.shape
    equal_loc = np.where(X[:, feature_idx] == feature_val)
    y_equal = y[equal_loc]
    equal_p = y_equal.size / n
    not_equal_p = 1 - equal_p
    not_equal_loc = np.where(X[:, feature_idx] != feature_val)
    y_not_equal = y[not_equal_loc]
    return equal_p * score_func(y_equal) + not_equal_p * score_func(y_not_equal)

# 计算连续特征下，按照某个值划分数据集得到的指标
def continuous_feature_partition(
        X: np.ndarray,
        y: np.ndarray,
        feature_idx: int,
        feature_val: float,
        score_func: Callable = mean_square_error
):
    n, _ = X.shape
    less_loc = np.where(X[:, feature_idx] <= feature_val)
    y_less = y[less_loc]
    less_p = y_less.size / n
    more_p = 1 - less_p
    more_loc = np.where(X[:, feature_idx] > feature_val)
    y_more = y[more_loc]
    return less_p * score_func(y_less) + more_p * score_func(y_more)


def select_best_feature_val(X: np.ndarray, y: np.ndarray, feature_idx: int, continuous: bool = True):
    feature_partition = discrete_feature_partition
    if continuous:
        feature_partition = continuous_feature_partition
    all_vals = X[:, feature_idx].reshape(-1)
    all_vals = np.unique(all_vals)
    if all_vals.size == 1:
        return all_vals[0], 0
    if continuous:
        all_vals = np.sort(all_vals)
        all_vals = (all_vals[:-1] + all_vals[1:]) / 2
    best_val = all_vals[0]
    best_score = inf
    # 对应gini或mse，分数越小越好
    for val in all_vals:
        score = feature_partition(X, y, feature_idx, val)
        if score < best_score:
            best_val = val
            best_score = score
    return best_val, best_score

# 从所有特征中选择最优的特征及对应的特征取值
def select_best_feature(X: np.ndarray, y: np.ndarray, feature_idx_list: List[int], feature_attr: List[str]):
    """
    :param X:
    :param y:
    :param feature_idx_list:
    :param feature_attr: 'discrete' or 'continuous'
    :return: best_feature_idx, best_val
    """
    # 遍历所有特征及对应的特征取值，计算gini指数
    best_feature_idx = 0
    best_feature_val = None
    best_feature_attr = None
    best_feature_score = inf
    for i in range(len(feature_idx_list)):
        feature_idx = feature_idx_list[i]
        flag = feature_attr[i] == "continuous"
        feature_val, feature_score = select_best_feature_val(X, y, feature_idx, flag)
        if feature_score < best_feature_score:
            best_feature_idx = feature_idx
            best_feature_val = feature_val
            best_feature_attr = feature_attr[i]
            best_feature_score = feature_score
    return best_feature_idx, best_feature_val, best_feature_attr, best_feature_score

@dataclass
class TreeNode:
    def __init__(
            self,
            feature_idx=-1,
            feature_val=-1,
            continuous=True,
            is_leaf_node=False,
            value=None
    ):
        """
        :param feature_idx: 特征id
        :param feature_val: 划分的特征值
        :param continuous: 是否是连续特征，默认为True
        :param is_leaf_node: 是否是叶节点
        :param value: 待预测的标签或值
        """
        self.feature_idx = feature_idx
        self.feature_val = feature_val
        self.continuous = continuous
        self.is_leaf_node = is_leaf_node
        self.value = value
        self.left_child = None
        self.right_child = None

    def decide_child(self, val):
        """
        根据值和特征是否连续决定分支
        :param val:
        :return:
        """
        if self.continuous:
            if val <= self.feature_val:
                return self.left_child
            else:
                return self.right_child
        else:
            if val == self.feature_val:
                return self.left_child
            else:
                return self.right_child



class CartTree:
    def __init__(self, regression=True, min_threshold=1, max_depth=10):
        self.min_threshold = min_threshold
        self.max_depth = max_depth
        self.root = None # 根节点
        self.feature_idx_list = None
        self.feature_attr = None
        self.regression = regression



    def build_tree(
            self,
            X: np.ndarray,
            y: np.ndarray,
            feature_idx_list: List[int],
            feature_attr: List[str],
            depth: int
    ) -> 'TreeNode':
        # 边界条件：最小样本数，所有类均相同，没有可划分的特征，达到最大深度
        num_sample = X.shape[0]
        cnt = Counter(y)
        num_class = len(cnt)
        value = cnt.most_common(1)[0][0]
        if self.regression:
            value = y.mean()
        # len(feature_idx_list) <= 1 or \
        if num_sample <= self.min_threshold or \
            num_class <= 1 or \
            depth >= self.max_depth:
            return TreeNode(is_leaf_node=True, value=value)
        best_idx, best_val, best_attr, best_score = \
            select_best_feature(X, y, feature_idx_list, feature_attr)
        node = TreeNode(best_idx, best_val, best_attr == "continuous",
                        is_leaf_node=False, value=value)
        if node.continuous:
            left_loc = np.where(X[:, best_idx] <= best_val)
            right_loc = np.where(X[:, best_idx] > best_val)
        else:
            left_loc = np.where(X[:, best_idx] == best_val)
            right_loc = np.where(X[:, best_idx] != best_val)
        X_left, y_left = X[left_loc], y[left_loc]
        X_right, y_right = X[right_loc], y[right_loc]
        node.left_child = self.build_tree(X_left, y_left, feature_idx_list, feature_attr, depth+1)
        node.right_child = self.build_tree(X_right, y_right, feature_idx_list, feature_attr, depth+1)
        return node


    def fit(self, X: np.ndarray, y: np.ndarray=None, feature_attr: List[str] = None):
        if y is None:
            y = X[:, -1]
            X = X[:, :-1]
        feature_cnt = X.shape[1]
        feature_idx_list = list(range(feature_cnt))
        if feature_attr is None:
            feature_attr = ["continuous"] * feature_cnt # 默认都是连续特征
        self.feature_idx_list = feature_idx_list
        self.feature_attr = feature_attr
        self.root = self.build_tree(X, y, feature_idx_list, feature_attr, 1)
        return self

    def _predict_single(self, x:np.ndarray, node: 'TreeNode'):
        if node.is_leaf_node:
            return node.value
        next_node = node.decide_child(x[node.feature_idx])
        return self._predict_single(x, next_node)

    def predict_single(self, x: np.ndarray):
        return self._predict_single(x, self.root)

    def predict(self, X: np.ndarray):
        y_pred = []
        for x in X:
            y_pred.append(self.predict_single(x))
        return np.array(y_pred)

    def _print_tree(self, node, depth):
        if node is None:
            return
        prefix = "----" * (depth - 1)
        if node.is_leaf_node:
            content = f"y: {node.value}"
        else:
            content = f"val <= {node.feature_val}"
        print(prefix + "|")
        print(prefix + content)
        self._print_tree(node.left_child, depth + 1)
        self._print_tree(node.right_child, depth+1)

    def print_tree(self):
        self._print_tree(self.root, 1)


if __name__ == "__main__":
    # 测试决策树
    # from sklearn.datasets import load_iris
    # X, y = load_iris(return_X_y=True)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
    # tree = CartTree()
    # tree.fit(X_train, y_train)
    # tree.print_tree()
    # y_pred = tree.predict(X_test)
    # acc = accuracy_score(y_pred, y_test)
    # print(acc)

    X = np.arange(1, 11).reshape(-1, 1)
    y = np.array([5.56, 5.70, 5.91, 6.40, 6.80, 7.05, 8.90, 8.70, 9.00, 9.05])
    tree = CartTree()
    tree.fit(X, y)
    tree.print_tree()