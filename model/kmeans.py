
from random import randint
import numpy as np

import torch


def eu_distance(x: np.ndarray, y: np.ndarray):
    """
    # 计算欧式距离，形状为n1, n2
    :param x: n1, d
    :param y: n2, d
    :return: dist
    """
    n1, _ = x.shape
    n2, _ = y.shape
    x = x.reshape((n1, 1, -1))
    y = y.reshape((1, n2, -1))
    e = x - y
    e = np.sum(e ** 2, axis=-1)
    return np.sqrt(e)



class KMeans:
    def __init__(self, num_clusters, max_iters=100, tol=1e-8):
        self.num_clusters = num_clusters
        self.max_iters = max_iters
        self.tol = tol
        self.centers = None
        self.label_ = None

    def init_centers_random(self, X: np.ndarray):
        # 随机初始化方法
        random_centers = np.random.choice(X, size=self.num_clusters, replace=False)
        return random_centers

    def init_centers_plusplus(self, X: np.ndarray):
        N, _ = X.shape
        first_idx = randint(0, N-1)
        centers = X[first_idx].reshape(1, -1)
        for i in range(1, self.num_clusters):
            dist = eu_distance(X, centers)
            min_dist = dist.min(axis=-1).squeeze()
            probs = min_dist / min_dist.sum()
            new_idx = np.random.choice(N, p=probs)
            centers = np.concatenate((centers, X[new_idx]))
        return centers

    def init_centers(self, X: np.ndarray, method:str="kmenas"):
        pass


    def assign(self, X):
        # 返回1个n的array
        dist = eu_distance(X, self.centers)
        assigned = dist.argmin(axis=-1)
        return assigned

    def update(self, X, assigned):
        new_centers = []
        for c in range(self.num_clusters):
            center = X[assigned==c].mean(axis=0).tolist()
            new_centers.append(center)
        return np.array(new_centers, dtype=X.dtype)

    def fit(self, X: np.ndarray):
        # 初始化聚类中心
        # 按照距离将样本分配到聚类中心
        # 更新聚类中心
        # 迭代前两步，直到收敛或达到最大迭代次数
        old_centers = self.init_centers(X)
        for _ in range(self.max_iters):
            assigned = self.assign(X)
            new_centers = self.update(X, assigned)
            if eu_distance(old_centers, new_centers).max() < self.tol:
                break
            old_centers = new_centers
        self.centers = old_centers
        self.label_ = self.assign(X)

    def predict(self, X):
        return self.assign(X)


if __name__ == "__main__":
    x = np.random.random((2, 4))
    y = np.random.random((3, 4))
    dist = eu_distance(x, y)
    print(dist.shape)
    import torch
    x = torch.tensor(x)
    y = torch.tensor(y)
    dist2 = torch.cdist(x, y).numpy()
    print(dist)
    print(dist2)