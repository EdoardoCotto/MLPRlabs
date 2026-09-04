import utils
import numpy as np
import scipy.linalg as lin
import matplotlib.pyplot as plt
import os


def compute_pca(D, m):
    mu = utils.v_col(D.mean(1))
    DC = D - mu
    C = DC @ DC.T / D.shape[1]

    _, U = np.linalg.eigh(C)
    P = U[:, ::-1][:, 0:m]
    return mu, P

def compute_lda(D, L):
    classes = np.unique(L)
    mu = utils.v_col(D.mean(axis=1))

    Sw = np.zeros((D.shape[0], D.shape[0]))
    Sb = np.zeros((D.shape[0], D.shape[0]))

    for label in classes:
        Dc = D[:, L == label]
        mu_c = utils.v_col(Dc.mean(axis=1))

        Dc_centered = Dc - mu_c
        Sw += Dc_centered @ Dc_centered.T

        diff = mu_c - mu
        Sb += Dc.shape[1] * (diff @ diff.T)

    Sw /= D.shape[1]
    Sb /= D.shape[1]

    eigenvalues, eigenvectors = lin.eigh(Sb, Sw)
    w = eigenvectors[:, -1:] 
    return w


def plot_results(
    scores_val,
    LVAL,
    threshold,
    accuracies,
    output_dir="outputs"
):
    os.makedirs(output_dir, exist_ok=True)

    predictions = np.where(scores_val > threshold, 2, 1)
    correct = predictions == LVAL

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Distribuzione degli score LDA
    axes[0].hist(
        scores_val[LVAL == 1],
        bins=10,
        alpha=0.7,
        label="Versicolor",
        color="royalblue"
    )
    axes[0].hist(
        scores_val[LVAL == 2],
        bins=10,
        alpha=0.7,
        label="Virginica",
        color="darkorange"
    )
    axes[0].axvline(
        threshold,
        color="black",
        linestyle="--",
        label=f"Soglia = {threshold:.2f}"
    )
    axes[0].set_title("Distribuzione degli score LDA")
    axes[0].set_xlabel("Score LDA")
    axes[0].set_ylabel("Numero di campioni")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Score della validation set
    sample_indices = np.arange(scores_val.size)
    axes[1].scatter(
        sample_indices[correct],
        scores_val[correct],
        color="seagreen",
        label="Corretto"
    )
    axes[1].scatter(
        sample_indices[~correct],
        scores_val[~correct],
        color="crimson",
        marker="x",
        s=70,
        label="Errato"
    )
    axes[1].axhline(threshold, color="black", linestyle="--")
    axes[1].set_title("Predizioni sulla validation set")
    axes[1].set_xlabel("Indice campione")
    axes[1].set_ylabel("Score LDA")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "lda_validation_results.png"),
        dpi=300
    )
    plt.close()

    # Accuracy al variare di m
    plt.figure(figsize=(7, 5))
    dimensions = np.arange(1, len(accuracies) + 1)

    plt.plot(
        dimensions,
        accuracies,
        marker="o",
        color="seagreen"
    )
    plt.xticks(dimensions)
    plt.ylim(0, 1.05)
    plt.xlabel("Dimensioni PCA")
    plt.ylabel("Accuracy")
    plt.title("Accuracy al variare della dimensionalità PCA")
    plt.grid(alpha=0.3)

    plt.savefig(
        os.path.join(output_dir, "accuracy_vs_pca_dimensions.png"),
        dpi=300
    )
    plt.close()

def classify(filename):
    dataset = utils.load_iris(filename)
    LIris = dataset[:, -1].astype(int)
    DIris = dataset[:, :-1].T

    #Only consider Versicolor and Virginica
    D = DIris[:, LIris != 0]
    L = LIris[LIris != 0]

    (DTR, LTR), (DVAL, LVAL) = utils.split_db_2to1(D,L)

    accuracies = []
    best_accuracy = -1
    best_results = None

    for m in range(1, DTR.shape[0] + 1):
        mu_pca, P = compute_pca(DTR, m)

        DTR_PCA = P.T @ (DTR - mu_pca)
        DVAL_PCA = P.T @ (DVAL - mu_pca)

        w = compute_lda(DTR_PCA, LTR)

        scores_train = (w.T @ DTR_PCA).ravel()
        scores_val = (w.T @ DVAL_PCA).ravel()

        mean_class_1 = scores_train[LTR == 1].mean()
        mean_class_2 = scores_train[LTR == 2].mean()

        if mean_class_1 > mean_class_2:
            scores_train = -scores_train
            scores_val = -scores_val

        threshold = (
            scores_train[LTR == 1].mean()
            + scores_train[LTR == 2].mean()
        ) / 2

        predictions = np.where(scores_val > threshold, 2, 1)
        accuracy = (predictions == LVAL).mean()
        accuracies.append(accuracy)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_results = scores_val.copy(), threshold

        print(
            f"PCA dimensions: {m}, "
            f"threshold: {threshold:.4f}, "
            f"accuracy: {accuracy:.4f}"
        )

    scores_val, threshold = best_results
    plot_results(scores_val, LVAL, threshold, accuracies)


    



if __name__ == "__main__":
    classify("iris.csv")