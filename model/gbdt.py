from typing import Optional, Union, List

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from cart import CartTree


class GBDTRegression:
    def __init__(
            self,
            base_model=CartTree,
            min_square_error=0.01,
            max_num_tree=5,
            **kwargs
    ):
        """
        :param base_model: 基础模型，默认是Cart回归树
        :param min_square_mean: gbdt参数，当均方根达到停止
        :param kwargs: base_model的输入参数
        """""
        self.trees: List[CartTree] = []
        self.min_square_error = min_square_error
        self.max_num_tree = max_num_tree
        self.base_model = base_model
        self.kwargs = kwargs
        self.init_mean_val = 0

    # 计算均方误差
    @staticmethod
    def calculate_square_error(y: np.ndarray, c: Optional[Union[float, np.ndarray]]=None):
        if c is None:
            c = y.mean()
        error = (y - c) ** 2
        return error.sum()

    def build_tree(self, X: np.ndarray, y: np.ndarray, **kwargs):
        # 初始化树
        tree = self.base_model(**self.kwargs)
        # 根据给定数据构建树
        tree.fit(X, y, **kwargs)
        # 添加
        self.trees.append(tree)
        # 计算针对X的输出
        y_pred = tree.predict(X)
        error = self.calculate_square_error(y, y_pred)
        new_y = y - y_pred
        return new_y, error

    def build(self, X: np.ndarray, y: np.ndarray, **kwargs):
        self.init_mean_val = y.mean()
        error = self.calculate_square_error(y, self.init_mean_val)
        y = y - self.init_mean_val
        while error > self.min_square_error and len(self.trees) < self.max_num_tree:
            y, error = self.build_tree(X, y, **kwargs)

    def fit(self, X: np.ndarray, y: Optional[np.ndarray]=None, **kwargs):
        if y is None:
            y = X[:, -1]
            X = X[:, :-1]
        self.build(X, y, **kwargs)

    def predict_single(self, x: np.ndarray):
        val = self.init_mean_val
        for tree in self.trees:
            val += tree.predict_single(x)
        return val

    def predict(self, X: np.ndarray):
        val_pred = []
        for x in X:
            val_pred.append(self.predict_single(x))
        return np.array(val_pred, dtype=np.float32)

if __name__ == "__main__":

    from sklearn.datasets import load_iris
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
    gbdt = GBDTRegression()
    gbdt.fit(X_train, y_train)
    y_pred = gbdt.predict(X_test)

    print(y_test[:10])
    print(y_pred[:10])
    print(len(gbdt.trees))
