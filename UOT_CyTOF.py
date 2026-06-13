import pandas as pd
import numpy as np
import phate
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import graphtools
from scipy.sparse.csgraph import connected_components
from sklearn.cluster import KMeans
import signal
import sys
import convolutional_sinkhorn
import matplotlib.pyplot as plt
from joblib import Parallel, delayed, parallel_backend
from scipy.sparse.linalg import eigsh 
from scipy.special import ive 
import UOT_convolutional_sinkhorn
import math


RESULT_FILE = '/home/x/hpc_geodesic_5_3_25/condition_matrix_1st_rep.csv'

def plot_transport_on_phate_thresholds(K_raw, UOT_K_raw, phate_data, sources, unique_sources, color_dict,
                                        t, L_n, reg_m, k, save_prefix=''):
    """
    For each threshold in [0.05, 0.10, ..., 1.0], plot both balanced and UOT
    transport plans side by side on the PHATE embedding.
    """
    def prepare_K(K_raw):
        K = np.copy(K_raw)
        K = np.maximum(K, 0)
        k_min, k_max = K.min(), K.max()
        if k_max > k_min:
            return (K - k_min) / (k_max - k_min), k_min, k_max
        return None, k_min, k_max

    K_norm, K_min, K_max = prepare_K(K_raw)
    UOT_norm, UOT_min, UOT_max = prepare_K(UOT_K_raw)

    if K_norm is None or UOT_norm is None:
        print("No transport variation in one or both plans, skipping.")
        return

    thresholds = np.arange(0.01, 0.2, 0.01)

    for threshold in thresholds:
        fig, axes = plt.subplots(1, 2, figsize=(13.5, 6))
        threshold_label = f'{threshold:.2f}'

        for ax, K_plot, K_min_val, K_max_val, label, letter in [
            (axes[0], K_norm,   K_min, K_max, 'Balanced', 'b)'),
            (axes[1], UOT_norm, UOT_min, UOT_max, 'UOT', 'c)'),
        ]:
            above_mask = K_plot > threshold
            rows, cols = np.where(above_mask)

            actual_threshold_val = threshold * (K_max_val - K_min_val) + K_min_val

            # All cells in grey
            ax.scatter(phate_data[:, 0], phate_data[:, 1],
                       s=3, alpha=0.3, c='grey', zorder=2)

            if len(rows) > 0:
                transported_source_idx = np.unique(rows)

                # Transported source cells in red
                ax.scatter(phate_data[transported_source_idx, 0],
                           phate_data[transported_source_idx, 1],
                           s=6, alpha=0.8, c='red', zorder=3,
                           label=f'Transported cells (p>{threshold})')

                # Red transport lines
                for i, j in zip(rows, cols):
                    alpha_val = K_plot[i, j]
                    alpha_rendered = 0.05 + 0.55 * (alpha_val - threshold) / (1.0 - threshold + 1e-8)
                    ax.plot(
                        [phate_data[i, 0], phate_data[j, 0]],
                        [phate_data[i, 1], phate_data[j, 1]],
                        color='red',
                        alpha=float(np.clip(alpha_rendered, 0.05, 1.0)),
                        linewidth=0.5,
                        zorder=1
                    )
                ax.legend(loc='upper right', fontsize=7, framealpha=0.8)
            else:
                ax.text(0.5, 0.5, 'No edges above threshold',
                        ha='center', va='center', transform=ax.transAxes, fontsize=10)

            ax.set_title(f'{letter} {label} | threshold={threshold_label} (t={t}, k={k}, reg_m={reg_m})')
            ax.set_xlabel('PHATE1')
            ax.set_ylabel('PHATE2')


        plt.tight_layout()
        plt.savefig(
            f'/Users/x/Documents/Cancer-Associated Fibroblasts Regulate Patient-Derived Organoid Drug Responses___/transport_phate_thresh{threshold_label}_t{t}_L{L_n}_regm{reg_m}_k{k}.png',
            dpi=350, bbox_inches='tight', transparent=True
        )
        plt.show()
        plt.close()


# Function to save results and exit on SIGTERM
def save_results_and_exit(signum, frame):
    print(f"\nReceived termination signal {signum}. Saving results and exiting...")
    np.savetxt(RESULT_FILE, condition_matrix, delimiter=',', header='m_0', comments='')
    sys.exit(0)


signal.signal(signal.SIGTERM, save_results_and_exit)
signal.signal(signal.SIGUSR1, save_results_and_exit)


def compute_chebychev_coeff_all(phi, tau, K):
    """Compute the K+1 Chebychev coefficients for our functions."""
    return 2 * ive(np.arange(0, K + 1), -tau * phi)


def generate_gaussian_points(data, n_points, mean=0, std=0.5):
    gaussian_points = []
    for point in data:
        gaussian_points.append(np.random.normal(loc=point, scale=std, size=(n_points, len(point))))
    return np.array(gaussian_points)


def sinkhorn_task(A, B, dfs, L, phi, coeff, t, k, total_num_cells):
    
    condition_A = dfs[A]
    condition_B = dfs[B]
    index_A = condition_A.index
    index_B = condition_B.index
    
    m_0 = np.zeros(total_num_cells[0])
    m_0[index_A[0]:index_A[-1]+1] = 1
    m_0 /= np.sum(m_0)
    
    m_1 = np.zeros(total_num_cells[0])
    m_1[index_B[0]:index_B[-1]+1] = 1
    m_1 /= np.sum(m_1)
    breakpoint()
    dist_w, K = convolutional_sinkhorn.fastcheb_conv_sinkhorn(L, m_0, m_1, phi, coeff, t=t, k=k, P=0)
    
    return (A, B, dist_w), K
    
    
def UOT_sinkhorn_task(A, B, dfs, L, phi, coeff, reg_m, t, k, total_num_cells):
    
    condition_A = dfs[A]
    condition_B = dfs[B]
    index_A = condition_A.index
    index_B = condition_B.index
    
    m_0 = np.zeros(total_num_cells[0])
    m_0[index_A[0]:index_A[-1]+1] = 1
    
    m_1 = np.zeros(total_num_cells[0])
    m_1[index_B[0]:index_B[-1]+1] = 1
    breakpoint()
    dist_w, K = UOT_convolutional_sinkhorn.fastcheb_conv_sinkhorn(L, m_0, m_1, phi, coeff, reg_m, t=t, k=k, P=1)
    
    return (A, B, dist_w), K


def build_affinity_matrix(O, N, sigma):
    dist_NO = pairwise_distances(N, O, squared=True)
    dist_ON = dist_NO.T
    K_NO = np.exp(-dist_NO / (2 * sigma ** 2))
    K_ON = np.exp(-dist_ON / (2 * sigma ** 2))
    K_NN = K_NO @ K_ON
    return K_NN


def sort_swiss(data, label):
    distances = np.linalg.norm(data, axis=1)
    sorted_idx = np.argsort(distances)
    return data[sorted_idx], label[sorted_idx]


def reference_selection(data, n_clusters, random_state):
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    kmeans.fit(data)
    return kmeans.cluster_centers_


def generate_gaussian(data, num_gauss_points, std=0.5):
    gaussian_points = []
    for point in data:
        points = np.random.normal(loc=point, scale=std, size=(num_gauss_points, len(point)))
        gaussian_points.append(points)
    return np.vstack(gaussian_points)


def magic(O, N, K_NN, t):
    D = np.diag(np.sum(K_NN, axis=1))
    P_NN = np.linalg.inv(D) @ K_NN
    P_NN_powered = np.linalg.matrix_power(P_NN, t)
    N_magic = P_NN_powered @ N
    all_points = np.vstack([O, N_magic])
    return N_magic, all_points


print('importing data...')

numeric_cols = ["pHistone_H2A", 'pCHK1', 'pDNAPK', 'cCaspase_3', 'cPARP']

data_centered = pd.read_csv('/Users/x/Documents/Cancer-Associated Fibroblasts Regulate Patient-Derived Organoid Drug Responses___/df_21_27_23.csv')
data_centered = data_centered[data_centered['source'].isin(['21_PDO_S_4_A_PDOs', '21_PDO_DMSO_0_A_PDOs'])]

source_rename = {
    '21_PDO_S_4_A_PDOs': 'SN-38',
    '21_PDO_DMSO_0_A_PDOs': 'DMSO'
}
data_centered['source'] = data_centered['source'].map(source_rename)

#data_centered = data_centered.drop(column='pPKCA')
# --- 20% subsample per source, preserving contiguous index structure ---
data_centered = (
    data_centered
    .groupby('source', group_keys=False)
    .apply(lambda grp: grp.sample(frac=0.2, random_state=42))
    .reset_index(drop=True)
)
print(f'Subsampled data shape: {data_centered.shape}')

dfs = {source: sub_df for source, sub_df in data_centered.groupby('source')}
for key, df in dfs.items():
    idx = df.index
    print(f"{key}: index range {idx[0]}–{idx[-1]}, length {len(idx)}, contiguous: {idx[-1] - idx[0] + 1 == len(idx)}")
breakpoint()
print(data_centered.columns)

numeric_cols = ['pHH3',  'EpCAM', 'CK18',
       'Pan_CK', 'IdU', 'pPDK1', 'cCaspase_3', 'Geminin', 'pMEK1_2',
       'pNDRG', 'pMKK4_SEK1', 'pBTK', 'pSRC', 'p4EBP1', 'pRB', 'pAKT308',
       'pCREB', 'pSMAD1_5_9', 'pAKT473', 'pNF_kB', 'pMKK3_MKK6', 'pP38',
       'pMAPKAPK', 'pAMPKa', 'pBAD', 'pHistone_H2A', 'p90RSK', 'pP120_catenin',
       'Beta_catenin_active', 'pGSK', 'pERK1_2', 'pSMAD2_3', 'PLK', 'CHGA',
       'pDNAPK', 'pS6', 'CD90', 'cPARP', 'pCHK1', 'Cyclin_B1']


#numeric_cols = data_centered.select_dtypes(include=[np.number]).columns
print('data imported')
print(data_centered.shape)
print(data_centered['source'].unique())


print('graph construction...')

band = 1.0
n_components = 10


phate_operator = phate.PHATE(random_state=42, t='auto', n_jobs=-1)
phate_data = phate_operator.fit_transform(data_centered[numeric_cols].to_numpy(dtype=np.float64))

print('phate calculation')
sources = data_centered['source'].values
counts = data_centered['source'].value_counts().to_dict()
unique_sources = np.unique(sources)
count_pdo1 = counts.get(unique_sources[0], 0)
count_pdo2 = counts.get(unique_sources[1], 0)
print(count_pdo1)
print(count_pdo2)

t = 'auto'

print('phate plotting...')
plt.figure(figsize=(8, 6))

color_dict = {unique_sources[0]: 'green', unique_sources[1]: 'blue'}
point_colors = [color_dict[src] for src in sources]
plt.scatter(phate_data[:, 0], phate_data[:, 1], s=10, alpha=0.7, c=point_colors)
handles = [plt.Line2D([0], [0], marker='o', color='w', label=src,
                      markerfacecolor=color_dict[src], markersize=5)
           for src in unique_sources]
plt.legend(handles=handles, loc='upper left', borderaxespad=0.5)
plt.title(f'a) DMSO and SN38 PDO data (t = {t})')
plt.savefig(f'/Users/x/Documents/Cancer-Associated Fibroblasts Regulate Patient-Derived Organoid Drug Responses___/phate_of_DSMO-SN38_20_08_25.png', dpi=400, bbox_inches='tight', transparent=True)
plt.show()

breakpoint()
graph = graphtools.Graph(data_centered[numeric_cols], use_pygsp=True, knn=20, n_jobs=-1)
graph.compute_laplacian("combinatorial")
L = graph.L
n_components, labels = connected_components(csgraph=L, directed=False, connection='weak')
print(f"\n\nThe (NO SUGAR) graph is composed of {n_components} connected component(s) bandwidth {band}.\n\n")




refined_data = data_centered[numeric_cols]

n_markers = len(refined_data.columns)
plots_per_figure = 9
n_figures = math.ceil(n_markers / plots_per_figure)

for fig_idx in range(n_figures):
    start_idx = fig_idx * plots_per_figure
    end_idx = min((fig_idx + 1) * plots_per_figure, n_markers)
    markers_in_figure = end_idx - start_idx
    
    n_cols = min(3, markers_in_figure)
    n_rows = math.ceil(markers_in_figure / n_cols)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    
    if markers_in_figure > 1:
        axes = axes.flatten()
    else:
        axes = [axes]
    
    for i, marker_idx in enumerate(range(start_idx, end_idx)):
        marker = refined_data.columns[marker_idx]
        ax = axes[i]
        
        sc = ax.scatter(
            phate_data[:, 0],
            phate_data[:, 1],
            c=refined_data[marker],
            cmap='viridis',
            s=5,
            alpha=0.7
        )
        ax.set_xlabel('PHATE1')
        ax.set_ylabel('PHATE2')
        ax.set_title(f'{marker}')
        cbar = fig.colorbar(sc, ax=ax, shrink=0.7)
        cbar.set_label(marker)
    
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()
    plt.savefig(
        f"/Users/x/Documents/Cancer-Associated Fibroblasts Regulate Patient-Derived Organoid Drug Responses___/phate_of_DSMO-SN38_20_08_25_ALL_fig{fig_idx + 1}.png",
        dpi=500, bbox_inches='tight', transparent=True
    )
    plt.show()


std = 2.5
sigma = 2.5

print('PCA plotting...')

data_centered.select_dtypes(include=['float64', 'int64']).columns

X = data_centered[numeric_cols]

t_list = [10.0]
k_list = [5]
knn_list = [10]
reg_m_list = [1.0]

markers = ['89Y_pHH3_S28_v7', '96Ru_96Ru', '98Ru_98Ru', '99Ru_99Ru',
           '100Ru_100Ru', '101Ru_101Ru', '102Ru_102Ru', '104Ru_104Ru',
           '111Cd_Vimentin RV202 (v67', '112Cd_FAP (1) v2',
           '113In_CD326 (EpCAM) (hu) (v6)', '114Cd_CK18 (v6)', '115In_Pan-CK_v9',
           '116Cd_GFP_v4', '127I_IdU', '142Nd_cCaspase 3_D175_v6', '143Nd_RRM2',
           '144Nd_SOX2 v2', '145Nd_pNDRG1 T346 v4', '146Nd_L1CAM', '147Sm_OPTN',
           '148Nd_CDK1 (1)', '149Sm_p4E-BP1_T37', '150Nd_pRB_S807_S811_v10',
           '151Eu_sqstm1', '153Eu_ANXA1', '155Gd_pAKT [S473] v12',
           '156Gd_pNF-kB p65 v8', '157Gd_MOPC21', '158Gd_pP38 MAPK v7',
           '160Gd_KI67(3)', '161Dy_pLATS1', '163Dy_H3K9Me3', '164Dy_TOP2A (3)',
           '165Ho_AlexaFluor488', '167Er_TROP 2(1)', '168Er_pSMAD2', '169Tm_EphB2',
           '170Er_CHGA v3', '171Yb_CD55 v4', '172Yb_BIRC3', '173Yb_pS6',
           '174Yb_cPARP [D214] (2) (v6)', '176Yb_CyclinB1 (2) (v7)', '191Ir_DNA 1',
           '193Ir_DNA 2', '209Bi_Me2HH3[K4]']


dfs = {source: sub_df for source, sub_df in data_centered.groupby('source')}
N = int(len(dfs))
total_num_cells = X.shape

L_list = [L]
L_n = 0
for L in L_list:
    for k in k_list:
        for reg_m in reg_m_list:
            t = 2.0

            phi = eigsh(L, k=k, return_eigenvectors=False)[0] / 2
            coeff = compute_chebychev_coeff_all(phi, t, k)
            print("Chebyshev coefficients:")
            print(coeff)

            condition_keys = list(dfs.keys())
            n_conditions = len(condition_keys)

            tasks = [(A, B)
                     for i, A in enumerate(condition_keys)
                     for B in condition_keys[i+1:]]

            print(f'Number of parallel tasks: {len(tasks)}')

            with parallel_backend("loky", inner_max_num_threads=1):
                parallel_results = Parallel(n_jobs=1, verbose=1)(
                    delayed(sinkhorn_task)(A, B, dfs, L, phi, coeff, t, k, total_num_cells)
                    for (A, B) in tasks
                )

            results = [item[0] for item in parallel_results]
            K = parallel_results[0][1] if parallel_results else None

            condition_matrix = np.zeros((n_conditions, n_conditions))

            for A, B, dist_w in results:
                i = condition_keys.index(A)
                j = condition_keys.index(B)
                condition_matrix[i, j] = dist_w
                condition_matrix[j, i] = dist_w

            tasks = [(A, B)
                     for i, A in enumerate(condition_keys)
                     for B in condition_keys[i+1:]]

            with parallel_backend("loky", inner_max_num_threads=1):
                UOT_parallel_results = Parallel(n_jobs=1, verbose=50)(
                    delayed(UOT_sinkhorn_task)(A, B, dfs, L, phi, coeff, reg_m, t, k, total_num_cells)
                    for (A, B) in tasks
                )

            UOT_results = [item[0] for item in UOT_parallel_results]
            UOT_K = UOT_parallel_results[0][1] if UOT_parallel_results else None

            UOT_condition_matrix = np.zeros((n_conditions, n_conditions))

            for A, B, UOT_dist_w in UOT_results:
                i = condition_keys.index(A)
                j = condition_keys.index(B)
                UOT_condition_matrix[i, j] = UOT_dist_w
                UOT_condition_matrix[j, i] = UOT_dist_w

            M = L.shape[0]
            a = np.ones(M) / M
            K_full = K * a
            UOT_K_full = UOT_K * a
            
            
            
            plt.figure(figsize=(8, 6))
            plt.imshow(K_full, cmap='viridis', interpolation='nearest', aspect='auto')
            plt.colorbar(label='Mass Transported')
            plt.title('Transport Plan')
            plt.xlabel('Cell Index')
            plt.ylabel('Cell Index')
            plt.tight_layout()
            plt.savefig(f'/Users/x/Documents/upgrade_UOT_graph_connection/Balanced_Transport_plan_18_08_25_t{t}_L{L_n}_regm{reg_m}_k{k}_1000.png', dpi=300, bbox_inches='tight', transparent=True)
            plt.show()
                

            plt.figure(figsize=(8, 6))
            plt.imshow(UOT_K_full, cmap='viridis', interpolation='nearest', aspect='auto')
            plt.colorbar(label='Mass Transported')
            plt.title('UOT Transport Plan')
            plt.xlabel('Cell Index')
            plt.ylabel('Cell Index')
            plt.tight_layout()
            plt.savefig(f'/Users/x/Documents/upgrade_UOT_graph_connection/UOT_Transport_plan_24_07_25_t{t}_L{L_n}_regm{reg_m}_k{k}_1000.png', dpi=300, bbox_inches='tight', transparent=True)
            plt.show()

            K_sliced = K_full[:500, :500]
            K_sliced = K_sliced[:100, 100:]
            UOT_K_sliced = UOT_K_full[:500, :500]
            UOT_K_sliced = UOT_K_sliced[:100, 100:]
            
            
            plt.figure(figsize=(8, 6))
            plt.imshow(K_sliced, cmap='viridis', interpolation='nearest', aspect='auto')
            plt.colorbar(label='Mass Transported')
            plt.title('Transport Plan')
            plt.xlabel('Cell Index')
            plt.ylabel('Cell Index')
            plt.tight_layout()
            plt.savefig(f'/Users/x/Documents/upgrade_UOT_graph_connection/Balanced_Transport_plan_18_08_25_t{t}_L{L_n}_regm{reg_m}_k{k}_1000.png', dpi=300, bbox_inches='tight', transparent=True)
            plt.show()
                

            plt.figure(figsize=(8, 6))
            plt.imshow(condition_matrix, cmap='viridis', interpolation='nearest', aspect='auto')
            plt.colorbar(label='Distance')
            plt.title('Condition Matrix')
            plt.xlabel('Condition Index')
            plt.ylabel('Condition Index')
            plt.tight_layout()
            plt.savefig(f'/Users/x/Documents/upgrade_UOT_graph_connection/Balanced_condition_matrix_24_07_25_t{t}_L{L_n}_regm{reg_m}_k{k}_1000.png', dpi=300, bbox_inches='tight', transparent=True)
            plt.show()

            plt.figure(figsize=(8, 6))
            plt.imshow(UOT_K_sliced, cmap='viridis', interpolation='nearest', aspect='auto')
            plt.colorbar(label='Mass Transported')
            plt.title('UOT Transport Plan')
            plt.xlabel('Cell Index')
            plt.ylabel('Cell Index')
            plt.tight_layout()
            plt.savefig(f'/Users/x/Documents/upgrade_UOT_graph_connection/UOT_Transport_plan_24_07_25_t{t}_L{L_n}_regm{reg_m}_k{k}_1000.png', dpi=300, bbox_inches='tight', transparent=True)
            plt.show()
                

            plt.figure(figsize=(8, 6))
            plt.imshow(UOT_condition_matrix, cmap='viridis', interpolation='nearest', aspect='auto')
            plt.colorbar(label='Distance')
            plt.title('UOT Condition Matrix')
            plt.xlabel('Condition Index')
            plt.ylabel('Condition Index')
            plt.tight_layout()
            plt.savefig(f'/Users/x/Documents/upgrade_UOT_graph_connection/UOT_condition_matrix_24_07_25_t{t}_L{L_n}_regm{reg_m}_k{k}_1000.png', dpi=300, bbox_inches='tight', transparent=True)
            plt.show()
            import pdb; pdb.set_trace()
            save_dir = '/Users/alistairwilkins/Documents/upgrade_UOT_graph_connection/'

            plot_transport_on_phate_thresholds(
                K_full, UOT_K_full, phate_data, sources, unique_sources, color_dict,
                t=t, L_n=L_n, reg_m=reg_m, k=k,
                save_prefix=save_dir
            )
            
'''

M = L.shape[0]
a = np.ones(M) / M
K = K * a
UOT_K = UOT_K * a

K = K[:500, :500]
K = K[:100, 100:]

UOT_K = UOT_K[:500, :500]
UOT_K = UOT_K[:100, 100:]

plt.figure(figsize=(8, 6))
plt.imshow(K, cmap='viridis', interpolation='nearest', aspect='auto')
plt.colorbar(label='Mass Transported')
plt.title('Transport Plan')
plt.xlabel('Cell Index')
plt.ylabel('Cell Index')
plt.tight_layout()
plt.savefig(f'/Users/x/Documents/upgrade_UOT_graph_connection/Balanced_Transport_plan_18_08_25_t{t}_L{L_n}_regm{reg_m}_k{k}_1000.png', dpi=300, bbox_inches='tight', transparent=True)
plt.show()
    

plt.figure(figsize=(8, 6))
plt.imshow(condition_matrix, cmap='viridis', interpolation='nearest', aspect='auto')
plt.colorbar(label='Distance')
plt.title('Condition Matrix')
plt.xlabel('Condition Index')
plt.ylabel('Condition Index')
plt.tight_layout()
plt.savefig(f'/Users/x/Documents/upgrade_UOT_graph_connection/Balanced_condition_matrix_24_07_25_t{t}_L{L_n}_regm{reg_m}_k{k}_1000.png', dpi=300, bbox_inches='tight', transparent=True)
plt.show()

plt.figure(figsize=(8, 6))
plt.imshow(UOT_K, cmap='viridis', interpolation='nearest', aspect='auto')
plt.colorbar(label='Mass Transported')
plt.title('UOT Transport Plan')
plt.xlabel('Cell Index')
plt.ylabel('Cell Index')
plt.tight_layout()
plt.savefig(f'/Users/x/Documents/upgrade_UOT_graph_connection/UOT_Transport_plan_24_07_25_t{t}_L{L_n}_regm{reg_m}_k{k}_1000.png', dpi=300, bbox_inches='tight', transparent=True)
plt.show()
    

plt.figure(figsize=(8, 6))
plt.imshow(UOT_condition_matrix, cmap='viridis', interpolation='nearest', aspect='auto')
plt.colorbar(label='Distance')
plt.title('UOT Condition Matrix')
plt.xlabel('Condition Index')
plt.ylabel('Condition Index')
plt.tight_layout()
plt.savefig(f'/Users/x/Documents/upgrade_UOT_graph_connection/UOT_condition_matrix_24_07_25_t{t}_L{L_n}_regm{reg_m}_k{k}_1000.png', dpi=300, bbox_inches='tight', transparent=True)
plt.show()

vmin = min(np.min(K), np.min(UOT_K))
vmax = max(np.max(K), np.max(UOT_K))

plt.figure(figsize=(8, 6))
plt.imshow(K, cmap='viridis', interpolation='nearest', aspect='auto', vmin =vmin, vmax = vmax)
plt.colorbar(label='Mass Transported')
plt.title('Transport Plan')
plt.xlabel('Cell Index')
plt.ylabel('Cell Index')
plt.tight_layout()
plt.savefig(f'/Users/x/Documents/upgrade_UOT_graph_connection/Scaled_Balanced_Transport_plan_18_08_25_t{t}_L{L_n}_regm{reg_m}_k{k}_1000.png', dpi=300, bbox_inches='tight', transparent=True)
plt.show()

plt.figure(figsize=(8, 6))
plt.imshow(UOT_K, cmap='viridis', interpolation='nearest', aspect='auto', vmin =vmin, vmax = vmax)
plt.colorbar(label='Mass Transported')
plt.title('UOT Transport Plan')
plt.xlabel('Cell Index')
plt.ylabel('Cell Index')
plt.tight_layout()
plt.savefig(f'/Users/x/Documents/upgrade_UOT_graph_connection/Scaled_UOT_Transport_plan_18_08_25_t{t}_L{L_n}_regm{reg_m}_k{k}_1000.png', dpi=300, bbox_inches='tight', transparent=True)
plt.show()

save_dir = '/Users/x/Documents/upgrade_UOT_graph_connection/'

plot_transport_on_phate(
    K_full, phate_data, sources, unique_sources, color_dict,
    t=t, L_n=L_n, reg_m=reg_m, k=k,
    label='Balanced',
    save_prefix=save_dir
)

plot_transport_on_phate(
    UOT_K_full, phate_data, sources, unique_sources, color_dict,
    t=t, L_n=L_n, reg_m=reg_m, k=k,
    label='UOT',
    save_prefix=save_dir
)
'''