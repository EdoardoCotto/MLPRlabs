import numpy as np
import utils
import matplotlib.pyplot as plt

def pca(filename, m):
    D = utils.load_iris(filename)
    mu, C = utils.get_mean_cov(D)

    L = D.T[-1, :]

    D = utils.center_data(D, mu)

    s, U = np.linalg.eigh(C)
    P = U[:, ::-1][:, 0:m]

    DP = np.dot(P.T, D)

    plt.figure(figsize=(7, 5))
    classes = [
        (0, "Iris-setosa", "blue"),
        (1, "Iris-versicolor", "orange"),
        (2, "Iris-virginica", "green")
    ]

    for label, name, color in classes:
        plt.scatter(
            DP[0, L == label], 
            DP[1, L == label], 
            label=name, 
            color=color,
            alpha=0.8
        )

    plt.xlabel("1st Principal Direction")
    plt.ylabel("2nd Principal Direction")
    plt.title("PCA - Iris Dataset (2D Projection)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.show()


if __name__ == "__main__":
    pca("iris.csv", m=2)