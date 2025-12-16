import numpy as np
from cart import CartTree


class GBDTRegression:
    def __init__(
            self,
            base_model=CartTree,
            min_square_mean=0.01,
            min_threshold=1,
            max_depth=10,
            max_num_tree=5
    ):
        """
        :param base_model: 基础模型，默认是Cart回归树
        :param min_square_mean: gbdt参数，当均方根达到停止
        :param min_threshold: cart参数，划分节点的最小样本数
        :param max_depth: cart参数，最大深度
        :param max_num_tree: 生成树的最大数量
        """""
        self.trees = []
        self.base_model = base_model
        self.min_square_mean = min_square_mean
        self.min_threshold = min_threshold
        self.max_depth = max_depth
        self.max_num_tree = max_num_tree


    def build_tree(self, X: np.ndarray, y: np.ndarray):
        pass

    def build(self):
        pass

    def fit(self):
        pass

    def predict(self):
        pass

