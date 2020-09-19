import scipy.sparse
import scipy
import numpy as np
import scipy as sp
import scipy.sparse as sprs
import scipy.spatial
import scipy.sparse.linalg

cimport numpy as np
cimport scipy as sp

DTYPE = np.float
ctypedef np.float_t DTYPE_t

cdef power_iterate(int N,
                   np.ndarray[DTYPE_t, ndim=2] M,
                   np.ndarray[DTYPE_t, ndim=2] x,
                   np.ndarray[DTYPE_t, ndim=1] p,
                   np.ndarray[DTYPE_t, ndim=1] dangling_weights, 
                   #np.ndarray[DTYPE_t, ndim=1] is_dangling, 
                   float alpha):

    cdef int i
    cdef int max_iter = 100
    cdef float tol = 1.0e-6
    #cdef np.ndarray[DTYPE_t, ndim=2] xlast = np.zeros(x.shape, dtype=np.float)
    cdef np.ndarray[DTYPE_t, ndim=2] xlast
    cdef float err

    # power iteration: make up to max_iter iterations
    for i in range(max_iter):
        xlast = x
        #x = alpha * (x * M + sum(x[is_dangling]) * dangling_weights) + \
        #    (1 - alpha) * p

        x = alpha * (np.dot(x, M) + dangling_weights) + \
            (1 - alpha) * p
        # check convergence, l1 norm
        x = x / x.sum()
        #err = scipy.absolute(x - xlast).sum()
        err = np.abs(x - xlast).sum()
        if err < N * tol:
            return x
    
    print('not convergence')

    return x

cdef linsolve(int N, 
              #sprs.csr_matrix[DTYPE_t] M,
              np.ndarray[DTYPE_t, ndim=2] x,
              np.ndarray[DTYPE_t, ndim=2] p,
              ):
    cdef sprs.csr_matrix I = sprs.eye(N)
    cdef np.ndarray[DTYPE_t] ppr

    ppr = sprs.sparse.linalg(I - alpha * M, (1 - alpha) * p)

    return ppr


def pagerank_cy(int N, 
                np.ndarray[DTYPE_t, ndim=2] M,
                np.ndarray[DTYPE_t, ndim=2] x,
                np.ndarray[DTYPE_t, ndim=2] p,
                np.ndarray[DTYPE_t, ndim=2] dangling_weights, 
                #np.ndarray[dtype_t, ndim=1] is_dangling, 
                float alpha):

    #print(M.shape)
    #print(x.shape)
    #print(p.shape)

    cdef int i
    cdef np.ndarray[DTYPE_t, ndim=2] ppr
    cdef np.ndarray[DTYPE_t, ndim=2] ppr_mat


    #ppr_mat = []
    #for i in range(p.shape[1]):
        #ppr = power_iterate(N, M, x, p[:, i], dangling_weights[:, i], is_dangling, 
                            #alpha, max_iter, tol)
        #ppr_mat.append(ppr)
        
    #return np.array(ppr_mat)

    for i in range(p.shape[1]):
        ppr = power_iterate(N, M, x, p[:, i], dangling_weights[:, i], 
                            alpha)
        ppr = linsolve(N, M, x, p[:, i], alpha)

        if i == 0:
            #ppr_mat = ppr[np.newaxis, :]
            ppr_mat = ppr
            #print(ppr_mat.shape)
        else:
            #ppr_mat = np.concatenate([ppr_mat, ppr[np.newaxis, :]])
            ppr_mat = np.concatenate([ppr_mat, ppr])
            #print(ppr_mat.shape)
        
    return ppr_mat