from typing import Sequence, List, Set, Any
from collections import Counter

import math
import numpy as np

import pandas as pd


def entropy(y: np.ndarray) -> float:
    # N,
    y = y.reshape((-1,))
    N = y.shape[0]
    cnt = np.bincount(y)
    pros = cnt / N
    res = np.sum([-p * np.log2(p) for p in pros if p > 0])
    return res.item()


# 计算在某个特征下的信息增益
def info_gain(X: np.ndarray, y: np.ndarray, feature_idx: int) -> float:
    N = y.size
    # 整体的信息熵
    total_ent = entropy(y)
    # 当前特征的单一值
    feature_unique = np.unique(X[:, feature_idx])
    # 针对每一种特征取值，计算个数与信息熵
    weighted_ent = 0.0
    for feature in feature_unique:
        sub_indices = np.where(X[:, feature_idx] == feature)
        sub_y = y[sub_indices]
        if sub_y.size > 0:
            p = sub_y.size / N
            weighted_ent += p * entropy(sub_y)
    res = total_ent - weighted_ent
    return res

# 从指定的特征集中选择最优的划分特征
def select_best_feature(
        X: np.ndarray,
        y: np.ndarray,
        feature_idx_set: Set[int],
        score_func=info_gain
):
    best_feature = None
    best_score = -math.inf
    for feature_idx in feature_idx_set:
        score = score_func(X, y, feature_idx)
        if score > best_score:
            best_feature = feature_idx
            best_score = score
    return best_feature, best_score


class Node:
    # 决策树节点
    def __init__(self, feature_idx=-1, threshold=-1, major=None, children=None, value=None, score=0.):
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.children = children
        self.value = value
        self.major = major
        self.score = score

    def is_leaf_node(self):
        return self.value is not None


# 基于numpy
class DecisionTree:
    def __init__(self, max_depth=10, min_sample_split=2, score_func=info_gain):
        self.max_depth = max_depth
        self.min_sample_split = min_sample_split
        self.root = None
        self.score_func = score_func
        self.feature_cnt = 0

    def build_tree(self, X: np.ndarray, y: np.ndarray, feature_idx_set: Set[int], depth: int = 0):
        """
        1. 判断终止条件：
            - 所有样本属于同一类别
            - 没有特征可用
            - 达到最少样本数
            - 达到最大深度
        2. 选择最优特征，并将其从特征集中去除
        3. 基于选出的最优特征的值创建节点，并划分数据集

        """
        N = y.size
        # 最少样本数
        cnt = Counter(y)
        major = cnt.most_common(1)[0][0]
        if N <= self.min_sample_split \
                or len(feature_idx_set) == 0 \
                or depth >= self.max_depth \
                or np.unique(y).size <= 1:
            return Node(value=major)
        # 选择最优特征
        best_feature, best_score = select_best_feature(X, y, feature_idx_set, score_func=self.score_func)
        feature_idx_set_new = feature_idx_set - {best_feature}
        # 特征取值
        best_feature_values = np.unique(X[:, best_feature])
        # 写入当前节点的major
        node = Node(feature_idx=best_feature, children=dict(), major=major, score=best_score)
        for value in best_feature_values:
            sub_idx = np.where(X[:, best_feature] == value)
            sub_X = X[sub_idx]
            sub_y = y[sub_idx]
            child = self.build_tree(sub_X, sub_y, feature_idx_set_new, depth+1)
            node.children[value] = child
        return node

    def fit(self, X, y=None, raw=False):
        # 如果y为None则默认X的最后一列为y
        if y is None:
            y = X[:, -1]
            X = X[:, :-1]
        self.feature_cnt = X.shape[1]
        feature_idx_set = set(range(X.shape[1]))
        self.root = self.build_tree(X, y, feature_idx_set)
        return self

    def predict_single(self, x, node):
        if node.is_leaf_node():
            return node.value
        feature_value = x[node.feature_idx]
        if feature_value in node.children:
            child_node = node.children[feature_value]
            return self.predict_single(x, child_node)
        else:
            return self.get_majority_class(node)

    def get_majority_class(self, node):
        return node.major

    def predict(self, X):
        X = np.array(X)
        pred = [self.predict_single(x, self.root).item() for x in X]
        return np.array(pred)

    def print_tree(self, node, feature_names: List[str]=None):
        if feature_names is None:
            feature_names = [str(i) for i in range(self.feature_cnt)]
        # 深度优先遍历

        pass



if __name__ == "__main__":
    # 测试决策树
    path = "../data/lenses/lenses.data"
    data = pd.read_csv(path, index_col=0, sep="\s+", header=None)
    data = np.array(data)
    X_train = data[:20, :-1]
    y_train = data[:20, -1]
    X_test = data[20:, :-1]
    y_test = data[20:, -1]

    dt = DecisionTree()
    dt.fit(X_train, y_train)
    y_pred = dt.predict(X_test)
    print(y_test)
    print(y_pred)
    # 寻找更丰富的决策树数据
    # 处理非数字输入
    # 实现cart
    from sklearn.tree import DecisionTreeClassifier, export_text
    sk_dt = DecisionTreeClassifier(criterion="entropy")
    sk_dt.fit(X_train, y_train)
    sk_y_pred = sk_dt.predict(X_test)
    print(sk_y_pred)
    tree_rule = export_text(sk_dt, feature_names=["a", "b", "c", "d"])
    print(tree_rule)