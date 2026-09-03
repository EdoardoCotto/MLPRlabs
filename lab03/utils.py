import pandas as pd
import numpy as np

def v_row(v):
    return np.reshape(v, (1, v.size))

def v_col(v):
    return np.reshape(v, (v.size, 1))

def load_iris(file_path):
    dic = {
        "Iris-setosa": 0,
        "Iris-versicolor": 1,
        "Iris-virginica": 2
    }

    df = pd.read_csv(file_path)
    target_col = df.columns[-1]
    df[target_col] = df[target_col].map(dic)

    matr = df.to_numpy()
    return matr

def center_data(data, mu):
    transposed = data.T
    centered = transposed[:-1, :] - mu
    return centered



def get_mean_cov(matr):
    transposed_matr = matr.T
    mu = transposed_matr[:-1, :].mean(1)
    mu = np.reshape(mu, (mu.size, 1))
    centered_matr = transposed_matr[:-1, :] - mu
    cov = 1 / centered_matr[0].size * centered_matr @ centered_matr.T
    return mu, cov
    


    


