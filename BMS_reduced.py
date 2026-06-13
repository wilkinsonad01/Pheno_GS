import time
import os
import matplotlib.pyplot as plt
import numpy as np
import phate
import pandas as pd
from scipy.sparse.linalg import eigsh
from scipy.special import ive
import graphtools
import scipy.sparse as sps
from sklearn.neighbors import kneighbors_graph
from joblib import Parallel, delayed
#import faiss

##################
##################
SOURCE = 'source'      #This needs to be adjusted!
##################
##################


COLUMN_BY_COLUMN_X2 = True


def expm_multiply(L, X, phi, coeff, K): 
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

def compute_chebychev_coeff_all(phi, tau, K):
    """Compute the K+1 Chebyshev coefficients for our functions."""
    return 2 * ive(np.arange(0, K + 1), -tau * phi)


def _process_column(j, L, M_v_col, phi, coeff, k_cheb, rows_per_cond, data_per_cond, pair_idx, n_cols, eps):
    """Compute X_2 for column j and all sum_M_w entries (i, j) for i > j."""
    X_2_col = expm_multiply(L, M_v_col, phi, coeff, k_cheb)
    results = []
    for i in range(j + 1, n_cols):
        rows_i = rows_per_cond[i]
        data_i = data_per_cond[i]
        if rows_i.size == 0:
            continue
        ratio = data_i / (X_2_col[rows_i] + eps)
        ratio_clipped = np.clip(ratio, eps, 1e32)
        contribution = (np.log(ratio_clipped + eps) * data_i).sum()
        results.append((pair_idx[i, j], contribution))
    return results

 
def main():
    t= 10.0
    k = 5
    knn= 10
    eps = 1e-8
    
    markers = ["pHH3", "EpCAM", "CK18", "Pan_CK", "IdU", "pPDK1", "cCaspase_3", "Geminin", "pMEK1_2", "pNDRG",
               "pMKK4_SEK1", "pBTK", "pSRC", "p4EBP1", "pRB", "pAKT308", "pCREB", "pSMAD1_5_9", "pAKT473", "pNF_kB",
               "pMKK3_MKK6", "pP38", "pMAPKAPK", "pAMPKa", "pBAD", "pHistone_H2A", "p90RSK", "pP120_catenin", "Beta_catenin_active", "pGSK", 
               "pERK1_2", "pSMAD2_3", "PLK", "CHGA", "pDNAPK", "pS6", "CD90", "cPARP", "pCHK1", "Cyclin_B1", "source"]

    df = pd.read_csv('/Users/x/Documents/HKWA_parallel/df_PDOs.csv')

    df = df[markers]
    df = df.reset_index(drop=True) 
    print(f"Final marker set: {len([m for m in markers if m != SOURCE])} features")

    print("\nTotal CPUs available:", os.cpu_count())

    data = df.copy()
    n_rows = len(df)
    data = data.drop([SOURCE], axis=1)
    dfs = {source: sub_df for source, sub_df in df.groupby(SOURCE)}
    N = len(dfs)
    print(f'N: {N}')

    graph = graphtools.Graph(data, use_pygsp=True,knn=knn, n_jobs=-1)  # Sharp distance decay (edges drop off quickly)
    graph.compute_laplacian("combinatorial")  # Compute normalized Laplacian
    L = graph.L

    graph_end_time = time.time()
    print("Normalised Laplacian built via FAISS.")


    phi = eigsh(L, k=k, return_eigenvectors=False)[0] / 2
    coeff = compute_chebychev_coeff_all(phi, t, k)
    condition_keys = list(dfs.keys())
    n_cols = N
    condition_ncells = {key: len(sub_df) for key, sub_df in dfs.items()}

    print('Building M ...')
    group_indices = {
        source: sub_df.index.values for source, sub_df in dfs.items()
    }
    starts, lengths = [], []
    for key in condition_keys:
        idx = group_indices[key]
        if len(idx) == 0:
            starts.append(0)
            lengths.append(0)
            continue
        starts.append(int(idx.min()))
        lengths.append(int(idx.max()) - int(idx.min()) + 1)
    lengths_arr = np.array(lengths, dtype=np.int64)
    nnz = int(lengths_arr.sum())

    row = np.empty(nnz, dtype=np.int32)
    col_ = np.empty(nnz, dtype=np.int32)
    vals = np.ones(nnz, dtype=bool)
    offset = 0
    for j, (start, length) in enumerate(zip(starts, lengths)):
        if length == 0:
            continue
        row[offset:offset + length] = np.arange(
            start, start + length, dtype=np.int32
        )
        col_[offset:offset + length] = j
        offset += length

    M = sps.csr_matrix((vals, (row, col_)), shape=(n_rows, n_cols))

    col_scales = np.array(
        [1.0 / condition_ncells[key] if condition_ncells[key] > 0 else 0.0
            for key in condition_keys],
        dtype=np.float64,
    )
    M = M.multiply(col_scales).tocsr()

    M_csc = M.tocsc()
    rows_per_cond = []
    data_per_cond = []
    for j in range(n_cols):
        s, e = M_csc.indptr[j], M_csc.indptr[j + 1]
        rows_per_cond.append(M_csc.indices[s:e])
        data_per_cond.append(M_csc.data[s:e])


    a = np.ones(n_rows, dtype=np.float64) / n_rows
    print('Computing X = exp(-tL) @ a ...')
    X = expm_multiply(L, a, phi, coeff, k) + eps

    print('Computing sum_M_v and M_v_dense ...')
    sum_M_v = np.zeros(n_cols, dtype=np.float64)
    M_v_dense = np.zeros((n_rows, n_cols), dtype=np.float64)
    for j in range(n_cols):
        rows_j = rows_per_cond[j]
        data_j = data_per_cond[j]
        if rows_j.size == 0:
            continue
        ratio = data_j / X[rows_j]       
        M_v_dense[rows_j, j] = ratio              
        ratio_clipped = np.clip(ratio, eps, 1e32)
        sum_M_v[j] = (np.log(ratio_clipped + eps) * data_j).sum()
    M_v_dense *= a[0]
    del X 


    n_pairs = n_cols * (n_cols - 1) // 2
    pair_idx = np.full((n_cols, n_cols), -1, dtype=np.int64)
    k_pair = 0
    for j in range(n_cols - 1):
        for i in range(j + 1, n_cols):
            pair_idx[i, j] = k_pair
            k_pair += 1

    sum_M_w = np.zeros(n_pairs, dtype=np.float64)

    if not COLUMN_BY_COLUMN_X2:
        print('Computing X_2 (batched, dense) ...')
        X_2 = expm_multiply(L, M_v_dense, phi, coeff, k)
        del M_v_dense

        print('Computing sum_M_w ...')
        for i in range(1, n_cols):
            rows_i = rows_per_cond[i]
            data_i = data_per_cond[i]
            if rows_i.size == 0:
                continue
            ratio = data_i[:, None] / (X_2[rows_i, :i] + eps)
            ratio_clipped = np.clip(ratio, eps, 1e32)
            contributions = np.log(ratio_clipped + eps) * data_i[:, None]
            sum_M_w[pair_idx[i, :i]] = contributions.sum(axis=0)

        del X_2
        
        
    else:
        print('Computing X_2 + sum_M_w column-by-column (parallel) ...')
        results_per_col = Parallel(n_jobs=-1, prefer='threads')(
            delayed(_process_column)(
                j, L, M_v_dense[:, j].copy(), phi, coeff, k,
                rows_per_cond, data_per_cond, pair_idx, n_cols, eps
            )
            for j in range(n_cols)
        )
        for col_results in results_per_col:
            for pair_index, contribution in col_results:
                sum_M_w[pair_index] = contribution
                

        del M_v_dense



    print('Assembling D_matrix ...')
    const = 4 * a[0] * t
    sum_M_v *= const
    sum_M_w *= const

    sum_M_v_ext = np.repeat(sum_M_v, np.arange(n_cols - 1, -1, -1))
    D_vec = sum_M_w + sum_M_v_ext

    D_matrix = np.zeros((n_cols, n_cols))
    upper_idx = np.triu_indices(n_cols, k=1)
    D_matrix[upper_idx] = D_vec
    D_matrix = D_matrix + D_matrix.T
    endtime = time.time()
    print(f"\nLaplacian computed in {graph_end_time - start_time:.2f} seconds")
    print(f"\nHKWA computed in {endtime - graph_end_time:.2f} seconds")
    print(f"\nCompleted in {endtime - start_time:.2f} seconds")
    print("\n── Exporting condition matrix ────────────────────────")
    breakpoint()
 
if __name__ == '__main__':
    main()