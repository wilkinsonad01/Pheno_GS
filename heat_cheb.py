import numpy as np
from scipy.sparse.linalg import eigsh  # Eigenvalues computation
from scipy.special import ive  # Bessel function
from scipy.special import factorial


def expm_multiply(L, X, phi, coeff,t, K, err): 
    inv_phi = 1.0 / phi
    two_inv_phi = 2.0 / phi

    T0 = X
    Y = 0.5 * coeff[0] * T0
    T1 = inv_phi * (L @ X) - T0
    Y = Y + coeff[1] * T1

    for j in range(2, K + 1):
        T2 = two_inv_phi * (L @ T1) - 2 * T1 - T0
        Y = Y + coeff[j] * T2
        T0 = T1
        T1 = T2

    return Y

