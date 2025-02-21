import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA


def plot_smartly(X, Y, Y_pred=None):
    """
    智能地绘制图形，支持输入多维的 X 和 Y 数组

    参数:
    X (numpy.array): 输入数据的 X 数组，形状为 (n, d)，n 表示样本数，d 表示特征数
    Y (numpy.array): 输入数据的 Y 数组，形状为 (n, 1)，n 表示样本数

    返回:
    无
    """
    n, d = X.shape

    # 判断 X 的特征数 d 是否大于等于 1
    if d < 1:
        raise ValueError("X 数组的特征数 d 必须大于等于 1")

    if d == 1:
        fig = plt.figure()

        sort_idx = np.argsort(X.flatten())

        # 使用索引对 X 和 Y 数组进行排序
        X = X[sort_idx]
        Y = Y[sort_idx]
        if Y_pred is not None:
            Y_pred = Y_pred[sort_idx]
        # 绘制一维折线图
        ax = fig.add_subplot(111)
        ax.plot(X, Y, '^', label='real', markersize=9, linestyle='-')
        if Y_pred is not None:
            ax.plot(X, Y_pred, 'o', label='predict',
                    markersize=3, linestyle='--')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        plt.legend()

    elif d == 2:

        angles = [30, 45, 60]
        # view_angles 是 angles 的笛卡尔积
        view_angles = [(elev, azim) for elev in angles for azim in angles]

        # 创建一个 3x3 的子图
        fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(
            10, 10), subplot_kw={'projection': '3d'})

        # 在每个子图中绘制一个不同视角的 3D 散点图
        for ax, view_angle in zip(axes.flat, view_angles):
            ax.scatter(X[:, 0], X[:, 1], Y, c='r', marker='o',
                       label='real', s=20, alpha=0.5)
            if Y_pred is not None:
                ax.scatter(X[:, 0], X[:, 1], Y_pred, c='b',
                           marker='o', label='predict', s=20, alpha=0.5)
            ax.view_init(elev=view_angle[0], azim=view_angle[1])
            # ax.set_title(f"View angle: {view_angle}")
            ax.legend()

        plt.tight_layout()

    else:
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)

        angles = [30, 45, 60]
        # view_angles 是 angles 的笛卡尔积
        view_angles = [(elev, azim) for elev in angles for azim in angles]

        # 创建一个 3x3 的子图
        fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(
            10, 10), subplot_kw={'projection': '3d'})

        # 在每个子图中绘制一个不同视角的 3D 散点图
        for ax, view_angle in zip(axes.flat, view_angles):
            ax.scatter(X_pca[:, 0], X_pca[:, 1], Y, c='r',
                       marker='o', label='real', s=20, alpha=0.5)
            if Y_pred is not None:
                ax.scatter(X_pca[:, 0], X_pca[:, 1], Y_pred, c='b',
                           marker='o', label='predict', s=20, alpha=0.5)
            ax.view_init(elev=view_angle[0], azim=view_angle[1])
            # ax.set_title(f"View angle: {view_angle}")
            ax.legend()

        plt.tight_layout()

    # 显示图
    plt.show()
