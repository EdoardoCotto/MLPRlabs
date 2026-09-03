import numpy as np
import scipy.linalg as lin
import utils
import matplotlib.pyplot as plt
import os

def compute_Sw_Sb(D, L, mu):
    N = D.shape[1]

    Sw = np.zeros((D.shape[0], D.shape[0]))
    Sb = np.zeros((D.shape[0], D.shape[0]))

    classes = np.unique(L)

    for c in classes:
        D_c = D[:, L == c]

        n_c = D_c.shape[1]
        mu_c = utils.v_col(D_c.mean(1))

        diff = mu - mu_c
        D_c_cen = D_c - mu_c

        Sb += n_c * np.dot(diff, diff.T)
        Sw += np.dot(D_c_cen, D_c_cen.T)

    return Sw / N, Sb / N



def lda(filename, m):
    out_dir = 'outputs'

    D = utils.load_iris(filename)
    mu, C = utils.get_mean_cov(D)

    L = D.T[-1, :]
    D = utils.center_data(D, mu)

    Sw, Sb = compute_Sw_Sb(D, L, np.zeros_like(mu))

    s, U = lin.eigh(Sb, Sw)
    W = U[:, ::-1][:, 0:m]

    D_lda = np.dot(W.T, D)

    save_path = os.path.join(out_dir, "lda_scatter_2d.png")

    plt.figure()
    labels = ["Iris-Setosa", "Iris-Versicolor", "Iris-Virginica"]
    for c in np.unique(L):
        plt.scatter(D_lda[0, L == c], D_lda[1, L == c], label=labels[int(c)])
    plt.xlabel("1st LDA direction")
    plt.ylabel("2nd LDA direction")
    plt.title("LDA - Iris Dataset (2D Projection)")
    plt.legend()
    plt.savefig(save_path, dpi=300)
    plt.close()



if __name__ == "__main__":
    lda('iris.csv', 2)
