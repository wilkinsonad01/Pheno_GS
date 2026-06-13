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
import faiss

##################
##################
SOURCE = 'source'      #This needs to be adjusted!
##################
##################


COLUMN_BY_COLUMN_X2 = True

def ask_yes_no(prompt):
    """Repeatedly ask a yes/no question until a valid answer is given."""
    valid = {'y', 'Y', 'n', 'N', 'yes', 'no', 'Yes', 'No', 'YES', 'NO', ''}
    while True:
        answer = input(prompt).strip()
        if answer in valid:
            return answer.lower() in {'y', 'yes', ''}
        print(f"'{answer}' is not a valid response. Please answer Y or n.")


def ask_numeric(prompt, default, cast=float, min_val=None, max_val=None):
    """
    Ask for a numeric value. Press Enter to accept the default.
    """
    while True:
        raw = input(f"{prompt} [default: {default}]: ").strip()
        if raw == '':
            return cast(default)
        try:
            value = cast(raw)
            if min_val is not None and value < min_val:
                print(f"Value must be >= {min_val}. Try again.")
                continue
            if max_val is not None and value > max_val:
                print(f"Value must be <= {max_val}. Try again.")
                continue
            return value
        except ValueError:
            print(f"'{raw}' is not a valid number. Try again.")


def ask_phate_t():
    """Ask for PHATE t — accepts a positive integer or 'auto'."""
    while True:
        raw = input("PHATE diffusion time t (positive integer or 'auto') [default: auto]: ").strip()
        if raw == '' or raw.lower() == 'auto':
            return 'auto'
        try:
            value = int(raw)
            if value < 1:
                print("t must be a positive integer. Try again.")
                continue
            return value
        except ValueError:
            print(f"'{raw}' is not valid. Enter a positive integer or 'auto'.")


def ask_graph_type():
    """Ask the user to choose between weighted (graphtools) or binary (sklearn) kNN graph."""
    options = {'1', '2', '3'}
    print("\n── Graph type ────────────────────────────────────────")
    print("  1) Weighted kNN  — graphtools (normalised Laplacian, distance-weighted edges)")
    print("  2) Binary kNN    — scikit-learn (connectivity matrix, unweighted edges)")
    print("  3) FAISS kNN    — FAISS (normalised Laplacian, distance-weighted edges)")
    while True:
        answer = input("  Select graph type [1/2/3, default: 3]: ").strip()
        if answer == '':
            return 3
        if answer in options:
            return int(answer)
        print(f"'{answer}' is not valid. Please enter 1, 2 or 3.")


def load_markers_from_file(filepath):
    try:
        with open(filepath, 'r') as f:
            markers = [line.strip() for line in f if line.strip()]
        if not markers:
            print(f"Marker file '{filepath}' is empty.")
            return None
        return markers
    except FileNotFoundError:
        print(f"File not found: '{filepath}'")
        return None
    except Exception as e:
        print(f"Could not read marker file: {e}")
        return None


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
    title = "Batched Matrix Geodesic Sinkhorn"
    print("\n" + "=" * 55)
    print(title.center(55))
    print("=" * 55)

    use_defaults = ask_yes_no("\nUse default settings? (t=10, k=5, knn=10, eps = 1e-8) [Y/n]: ")

    if use_defaults:
        t= 10.0
        k = 5
        knn= 10
        eps = 1e-8
        print(f"Using defaults: t={t}, k={k}, knn={knn}, eps = {eps}")
    else:
        print("\nEnter parameters (press Enter to keep the default):")
        t = ask_numeric("  Diffusion time (t)", default=10.0, cast=float, min_val=0.01)
        k = ask_numeric("  Chebyshev polynomial degree (k)",default=5, cast=int, min_val=1)
        knn = ask_numeric("  kNN parameter", default=10,cast=int, min_val=1)
        eps = ask_numeric("  eps parameter", default=1e-8,cast=float, min_val=1e-12)
 
    print("\n── Marker selection ──────────────────────────────────")
    use_marker_file = ask_yes_no("Load markers from an external .txt file? [Y/n]: ")

    markers = None

    if use_marker_file:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        default_marker_path = os.path.join(script_dir, 'markers.txt')

        if os.path.isfile(default_marker_path):
            print(f"Found 'markers.txt' in script directory: {default_marker_path}")
            use_default = ask_yes_no("  Use this file? [Y/n]: ")
            if use_default:
                markers = load_markers_from_file(default_marker_path)
                if markers is not None:
                    print(f"Loaded {len(markers)} markers.")
        else:
            print(f"No 'markers.txt' found in script directory ({script_dir}).")

        if markers is None:
            while True:
                marker_path = input("  Enter path to marker file: ").strip()
                markers = load_markers_from_file(marker_path)
                if markers is not None:
                    print(f"Loaded {len(markers)} markers from '{marker_path}'.")
                    confirm = ask_yes_no(f"  Use these {len(markers)} markers? [Y/n]: ")
                    if confirm:
                        break
                    else:
                        retry = ask_yes_no("  Try a different file? [Y/n]: ")
                        if not retry:
                            markers = None
                            break
                else:
                    retry = ask_yes_no("  Try a different file path? [Y/n]: ")
                    if not retry:
                        markers = None
                        break

    if markers is None:
        print("Will use all available features from the data.")

    graph_type = ask_graph_type()
    graph_label = (
        "Weighted kNN (graphtools)" if graph_type == 1
        else "Binary kNN (scikit-learn)" if graph_type == 2
        else "FAISS kNN (FAISS)"
    )
    print(f"Using: {graph_label}")

    print("\n── Output options ────────────────────────────────────")
    plot_condition_matrix = ask_yes_no("Plot the condition matrix when complete? [Y/n]: ")

    run_phate = ask_yes_no("Run PHATE on the condition matrix? [Y/n]: ")

    phate_t = None
    if run_phate:
        phate_t = ask_phate_t()
        print(f"PHATE will use t={phate_t}")
        
    print("\n── Loading data ──────────────────────────────────────")
    script_dir = os.path.dirname(os.path.abspath(__file__))


    csv_files = [f for f in os.listdir(script_dir) if f.endswith('.csv')]

    file_path = None

    if csv_files:
        if len(csv_files) == 1:
            candidate = os.path.join(script_dir, csv_files[0])
            print(f"Found data file in script directory: {candidate}")
            if ask_yes_no("  Use this file? [Y/n]: "):
                file_path = candidate
        else:
            # Multiple CSVs — list them and let the user pick
            print("Multiple CSV files found in script directory:")
            for i, f in enumerate(csv_files, start=1):
                print(f" {i}) {f}")
            while file_path is None:
                raw = input(f"  Select file [1–{len(csv_files)}, or press Enter to enter path manually]: ").strip()
                if raw == '':
                    break  # fall through to manual entry
                try:
                    idx = int(raw) - 1
                    if 0 <= idx < len(csv_files):
                        file_path = os.path.join(script_dir, csv_files[idx])
                    else:
                        print(f"Please enter a number between 1 and {len(csv_files)}.")
                except ValueError:
                    print(f"{raw}' is not valid. Enter a number or press Enter.")
    else:
        print(f"No CSV files found in script directory ({script_dir}).")


    if file_path is None:
        while True:
            raw = input("  Enter full path to data file: ").strip()
            if os.path.isfile(raw):
                file_path = raw
                break
            print(f"File not found: '{raw}'. Please check the path and try again.")

    df = pd.read_csv(file_path)
    #breakpoint()
    print(f"Loaded data: {df.shape[0]} cells X {df.shape[1]} features")

    if markers is None:
        markers = [c for c in df.columns if c != SOURCE] + [SOURCE]
        print(f"Using all {len(markers) - 1} features (+source column).")

    missing = [m for m in markers if m not in df.columns]
    if missing:
        print(f"\n Warning: {len(missing)} marker(s) not found in data and will be skipped:")
        for m in missing: 
            print(f"{m}")
        markers = [m for m in markers if m in df.columns]
    
    df = df[markers]
    print(f"Final marker set: {len([m for m in markers if m != SOURCE])} features")

    print("\nTotal CPUs available:", os.cpu_count())

    data = df.copy()
    n_rows = len(df)
    data = data.drop([SOURCE], axis=1)
    dfs = {source: sub_df for source, sub_df in df.groupby(SOURCE)}
    N = len(dfs)
    print(f'N: {N}')


    for _, sub_df in dfs.items():
        sub_df = sub_df.drop([SOURCE], axis=1)

    print(f"\n── Graph construction ({graph_label}) ────────────")

    if graph_type == 1:
        print('Timer started!')
        print("Calculating Laplacian...")
        start_time = time.time()
        graph = graphtools.Graph(data, use_pygsp=True, knn=knn)
        graph.compute_laplacian("normalized")
        L = graph.L
        graph_end_time = time.time()
        print("Normalised Laplacian built via graphtools.")
    elif graph_type == 3:
        C = ask_numeric("  efConstruction (C)", default=200.0, cast=int, min_val=10)
        S = ask_numeric("  efSearch (S)", default=100.0, cast=int, min_val=10)
        print('Timer started!')
        print("Calculating Laplacian...")
        start_time = time.time()
        Q, d = data.shape
        index = faiss.IndexHNSWFlat(d, 50) # max neighbors per node in the graph (higher = more accurate, more memory)
        index.hnsw.efConstruction = C # how thoroughly to search when building the graph (higher = better graph, slower build)
        index.hnsw.efSearch = S # = how thoroughly to search when querying (higher = more accurate results, slower queries)
        index.add(data)
        _, indices = index.search(data, knn + 1)
        indices = indices[:, 1:]

        rows = np.repeat(np.arange(Q), knn)
        cols = indices.reshape(-1)
        vals = np.ones(Q * knn, dtype=np.float32)

        A = sps.csr_matrix((vals, (rows, cols)), shape=(Q, Q))
        A = A.maximum(A.T)

        degrees = A.sum(axis=1).A1
        D = sps.diags(degrees, format="csr")
        L = D - A
        graph_end_time = time.time()
        print("Normalised Laplacian built via FAISS.")

    
    else:
        print('Timer started!')
        print("Calculating Laplacian...")
        start_time = time.time()
        A_graph = kneighbors_graph(data.values, n_neighbors=knn, mode='connectivity', include_self=False, n_jobs=-1)
        n, m = A_graph.shape
        diags = A_graph.sum(axis=1)
        D = sps.spdiags(diags.flatten(), [0], m, n, format='csr')
        L = D - A_graph
        graph_end_time = time.time()
        print("Unnormalised Laplacian built via scikit-learn.")
       

    print("\n── BMS ───────────────────────────────────────────")
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
    print(f"\nBMS computed in {endtime - graph_end_time:.2f} seconds")
    print(f"\nCompleted in {endtime - start_time:.2f} seconds")
    print("\n── Exporting condition matrix ────────────────────────")
    output_path = os.path.join(script_dir, 'condition_matrix.csv')
    D_df = pd.DataFrame(D_matrix, index=condition_keys, columns=condition_keys)
    D_df.to_csv(output_path)
    print(f"Saved condition matrix to: {output_path}")
    
    if plot_condition_matrix:
        print("\n── Plotting condition matrix ─────────────────────────")
        plt.figure(figsize=(8, 6))
        plt.imshow(D_matrix, cmap='viridis', interpolation='nearest')
        plt.colorbar(label='Distance')
        plt.title('Condition Matrix')
        plt.xlabel('Condition Index')
        plt.ylabel('Condition Index')
        plt.tight_layout()
        plt.show()

    if run_phate:
        print("\n── PHATE embedding ───────────────────────────────────")
        phate_operator = phate.PHATE(random_state=42, knn_dist='precomputed', mds_dist='minkowski',t=phate_t, mds_solver='smacof', n_jobs=-1,verbose=True)
        phate_operator.fit(D_matrix)
        data_phate = phate_operator.transform(D_matrix)
        print(f"PHATE embedding shape: {data_phate.shape}")
        phate_output_path = os.path.join(script_dir, 'phate_embedding.csv')
        phate_df = pd.DataFrame(data_phate, index=condition_keys, columns=[f'PHATE_{i+1}' for i in range(data_phate.shape[1])])
        phate_df.to_csv(phate_output_path)
        print(f"Saved PHATE embedding to: {phate_output_path}")
        
    print("\n── Done ──────────────────────────────────────────────\n")


if __name__ == '__main__':
    main()