import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA


def plot_smartly(X, Y, Y_pred=None):
    """
    X (numpy.array):(n, d)
    Y (numpy.array):(n, 1)
    """
    
    n, d = X.shape

    if d < 1:
        raise ValueError("The number of features d of the X array must be greater than or equal to 1")

    if d == 1:
        fig = plt.figure()

        sort_idx = np.argsort(X.flatten())

        X = X[sort_idx]
        Y = Y[sort_idx]
        if Y_pred is not None:
            Y_pred = Y_pred[sort_idx]

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

        view_angles = [(elev, azim) for elev in angles for azim in angles]

        fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(
            10, 10), subplot_kw={'projection': '3d'})

        for ax, view_angle in zip(axes.flat, view_angles):
            ax.scatter(X[:, 0], X[:, 1], Y, c='r', marker='o',
                       label='real', s=20, alpha=0.5)
            if Y_pred is not None:
                ax.scatter(X[:, 0], X[:, 1], Y_pred, c='b',
                           marker='o', label='predict', s=20, alpha=0.5)
            ax.view_init(elev=view_angle[0], azim=view_angle[1])

            ax.legend()

        plt.tight_layout()

    else:
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)

        angles = [30, 45, 60]

        view_angles = [(elev, azim) for elev in angles for azim in angles]

        fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(
            10, 10), subplot_kw={'projection': '3d'})
        
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

    plt.show()
