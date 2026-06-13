import heat_cheb as hc
from scipy.sparse.linalg import eigsh
import numpy as np
import pygsp
import warnings


def fastcheb_conv_sinkhorn(
    L,
    m_0,
    m_1,
    phi,
    coeff,
    reg_m,
    t=50, 
    k=10,
    err=1e-32,
    stopThr=1e-4,
    max_iter=1e3, #1e3,
    verbose=False,
    P=None,
    eps = 1e-8,
    **kwargs,
):

    N = L.shape[0]
    # G = pygsp.graphs.Graph(W)
    v = np.ones(N)
    w = np.ones(N)
    a = np.ones(N) / N
    if isinstance(reg_m, (int, float)):
        reg_m1, reg_m2 = get_parameter_pair(reg_m)
        
    fi_1 = reg_m1 / (reg_m1 + t) if isinstance(reg_m, (int, float)) else 1.0 
    fi_2 = reg_m2 / (reg_m2 + t) if isinstance(reg_m, (int, float)) else 1.0

    for i in range(1, int(max_iter) + 1):
        v_prev = v
        w_prev = w
        v = (m_0 / (hc.expm_multiply(L, a * w, phi, coeff, t, k, err) + eps)) #** fi_1
        w = (m_1 / (hc.expm_multiply(L, a * v, phi, coeff, t, k, err) + eps)) #** fi_2\
        v = np.clip(v, eps, 1e32)
        w = np.clip(w, eps, 1e32)
        v = v ** fi_1
        w = w ** fi_2


        if (
            np.any(np.isnan(v))
            or np.any(np.isnan(w))
            or np.any(np.isinf(v))
            or np.any(np.isinf(w))
         
        ):
            v = v_prev
            w = w_prev
            # we have reached the machine precision or negative value
            # come back to previous solution and quit loop
            print(
                f"Warning: numerical errors at iteration {i} cheb with t {t} and k {k}"
            )
            break
        if i % 10 == 0:
            if verbose: 
                print(i, np.sum(np.abs(v - v_prev)))
            if np.sum(np.abs(v - v_prev)) < stopThr:
                if verbose:
                    print("converged at iteration %d" % i)
                break
    if P is None:
        #print(np.sum(4 * a * t * (m_0 * np.log(v + eps) + m_1 * np.log(w + eps))))
        return np.sum(4 * a * t * (m_0 * np.log(v + eps) + m_1 * np.log(w + eps)))
    else:
        dist = np.sum(4 * a * t * (m_0 * np.log(v + eps) + m_1 * np.log(w + eps)))
        I = np.eye(L.shape[0])
        K_heat = hc.expm_multiply(L, I, phi, coeff, t, k, err)
        #print('heat kernel approximation used')
        #print(K_heat.shape)

        Plan = np.diag(v) @ K_heat @ np.diag(w)
        return dist, Plan
                
   
def get_parameter_pair(parameter):

    if isinstance(parameter, float) or isinstance(parameter, int):
        param_1, param_2 = parameter, parameter
    elif len(parameter) == 1:
        param_1, param_2 = parameter[0], parameter[0]
    else:
        if len(parameter) > 2:
            raise ValueError(
                "Parameter must be either a scalar, \
                             or an indexable object of length 1 or 2."
            )
        else:
            param_1, param_2 = parameter[0], parameter[1]

    return param_1, param_2