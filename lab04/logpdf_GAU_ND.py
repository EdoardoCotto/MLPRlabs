import numpy as np

def logpdf_GAU_ND(x, mu, C):
    M = mu.shape[0]
    log_det = np.linalg.slogdet(C)[1]
    C_inv = np.linalg.inv(C)

    return -M/2 * np.log(2*np.pi) - 1/2*log_det -1/2*(x - mu) @ C_inv @ ((x-mu).T)

