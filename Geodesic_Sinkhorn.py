import time
start_time = time.time()
import signal
import sys
import os
import convolutional_sinkhorn
import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed, parallel_backend
import pandas as pd
from scipy.sparse.linalg import eigsh 
from scipy.special import ive 
import graphtools


RESULT_FILE = '/home/x/hpc_geodesic_5_3_25/condition_matrix_1st_rep.csv'

def save_results_and_exit(signum, frame):
    print(f"\nReceived termination signal {signum}. Saving results and exiting...")
    np.savetxt(RESULT_FILE, condition_matrix, delimiter=',', header='m_0', comments='')
    sys.exit(0)  # Ensure graceful exit


signal.signal(signal.SIGTERM, save_results_and_exit)  
signal.signal(signal.SIGUSR1, save_results_and_exit)  


def compute_chebychev_coeff_all(phi, tau, K):
    """Compute the K+1 Chebychev coefficients for our functions."""
    return 2 * ive(np.arange(0, K + 1), -tau * phi)


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
    m_1 /=np.sum(m_1)
    dist_w = convolutional_sinkhorn.fastcheb_conv_sinkhorn(L, m_0, m_1, phi, coeff, t=t, k=k, P=None)
    return (A, B, dist_w)
    

def main():
    global condition_matrix 
    print("Total CPUs available:", os.cpu_count())

    markers = ['pHH3','EpCAM', 'CK18', 'Pan_CK', 
       'IdU', 'pPDK1', 'cCaspase_3', 'Geminin', 'pMEK1_2', 'pNDRG',
       'pMKK4_SEK1', 'pBTK', 'pSRC', 'p4EBP1', 'pRB', 'pAKT308', 'pCREB',
       'pSMAD1_5_9', 'pAKT473', 'pNF_kB', 'pMKK3_MKK6', 'pP38', 'pMAPKAPK',
       'pAMPKa', 'pBAD', 'pHistone_H2A', 'p90RSK', 'pP120_catenin',
       'Beta_catenin_active', 'pGSK', 'pERK1_2', 'pSMAD2_3', 'PLK', 'CHGA',
       'pDNAPK', 'pS6', 'CD90', 'cPARP', 'source']

    t_list = [10.0] 
    k_list = [5] 
    knn_list = [10] 


    file_path = '/Users/x/Documents/Cancer-Associated Fibroblasts Regulate Patient-Derived Organoid Drug Responses___/df_21.csv'
    df = pd.read_csv(file_path) 
    df = df[markers]
    data =df.copy()
    total_num_cells = data.shape[0]
    data = data.drop(['source'], axis=1)
    dfs = {source: sub_df for source, sub_df in df.groupby('source')}
    N = int(len(dfs)) 
    total_num_cells= df.shape

    for _, sub_df in dfs.items():
        sub_df = sub_df.drop(['source'], axis=1)

        
    for knn in knn_list:

        print('graph construction')
        #A) Epsilon Radius Graph Laplacian
        graph = graphtools.Graph(data, use_pygsp=True, knn=knn)
        graph.compute_laplacian("normalized")  # Compute normalized Laplacian
        L = graph.L

    
        #B) KNN Graph Laplacian 
        '''
        A = kneighbors_graph(data.values, n_neighbors=knn, mode='connectivity', include_self=False, n_jobs=-1)  # Sparse adjacency matrix
        n, m = A.shape
        diags = A.sum(axis=1)
        D = sps.spdiags(diags.flatten(), [0], m, n, format='csr')
        L = D - A  
        '''
        

        print('sinkhorn iterations')
        for k in k_list:
            for t in t_list:
                condition_matrix = np.zeros((N,N))  
                phi = eigsh(L, k=k, return_eigenvectors=False)[0] / 2  
                coeff = compute_chebychev_coeff_all(phi, t, k)  
                condition_keys = list(dfs.keys())
                n_conditions = len(condition_keys)
                
                tasks = [(A,B)
                            for i, A in enumerate(condition_keys)
                            for B in condition_keys[i+1:] 
                            ]
                
                with parallel_backend("loky", inner_max_num_threads=1):
                    results = Parallel(n_jobs=-1, verbose=1)(
                        delayed(sinkhorn_task)(A, B, dfs, L, phi, coeff, t, k, total_num_cells)
                        for (A, B) in tasks
                    )
                        
                condition_matrix = np.zeros((n_conditions,n_conditions))
                for A,B, dist_w in results:
                    An = list(dfs.keys()).index(A)
                    Bn = list(dfs.keys()).index(B)
                    condition_matrix[An,Bn] = dist_w
                    condition_matrix[Bn,An] = dist_w
                    
                endtime = time.time()
                print(f"{endtime - start_time:.2f} seconds")
                
                plt.figure(figsize=(8, 6))
                plt.imshow(condition_matrix, cmap='viridis', interpolation='nearest')
                plt.colorbar(label='Distance')
                plt.title('Condition Matrix')
                plt.xlabel('Condition Index')
                plt.ylabel('Condition Index')
                plt.tight_layout()
                plt.show()
                
                np.savetxt('/Users/x/Documents/Cancer-Associated Fibroblasts Regulate Patient-Derived Organoid Drug Responses___/GS_Trellis_distance_matrix_84.csv', condition_matrix, delimiter=',', comments='')
        


if __name__ == '__main__':
    main()