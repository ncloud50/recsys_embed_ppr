import networkx as nx
from networkx.exception import NetworkXError
import numpy as np
import scipy.sparse
import time


from fast_pagerank import pagerank
from fast_pagerank import pagerank_power

from scikits import umfpack
from sknetwork.ranking import PageRank
import scipy.sparse.linalg as linalg

import warnings
warnings.filterwarnings('ignore')

def pagerank_scipy(G, personal_vec=None, alpha=0.85, beta=0.01,
                   max_iter=100, tol=1.0e-6, weight='weight',
                   dangling=None,
                   cy=None):
    
    N = len(G)
    if N == 0:
        return {}

    nodelist = G.nodes()
    M = nx.to_scipy_sparse_matrix(G, nodelist=nodelist, weight=weight,
                                  dtype=float)
    S = scipy.array(M.sum(axis=1)).flatten()
    S = scipy.array(M.sum(axis=1)).flatten()
    S[S != 0] = 1.0 / S[S != 0]
    Q = scipy.sparse.spdiags(S.T, 0, *M.shape, format='csr')
    M = Q * M

    
    # initial vector
    x = scipy.repeat(1.0 / N , N) #np.ndarray

    # Personalization vector
    p = personal_vec #np.ndarray
 

    dangling_weights = p
    is_dangling = scipy.where(S == 0)[0] # np.ndarray
    #print(dangling_weights.shape)
    #print(is_dangling.shape)
    #print(is_dangling)

    #print(x.shape)
    #print(M.shape)
    #print(p.shape)


    if cy:
        import iterate_cy
        #import sys
        #sys.exit()
        #print(dangling_weights[:, 1].shape)
        ppr_mat = iterate_cy.pagerank_cy(N, 
                                        M.todense(), 
                                        x[np.newaxis, :], 
                                        p,
                                        dangling_weights, 
                                        #is_dangling, 
                                        alpha) 
        
    
    else:

        for i in range(p.shape[1]):
            ppr = power_iterate(N, M, x, p[:, i], dangling_weights[:, i], is_dangling, 
                                alpha, max_iter, tol)
            #print(ppr.shape)
            if i == 0:
                ppr_mat = ppr[np.newaxis, :]
                #print(ppr_mat.shape)
            else:
                ppr_mat = np.concatenate([ppr_mat, ppr[np.newaxis, :]])
                #print(ppr_mat.shape)
        
    return ppr_mat
    

def power_iterate(N, M, x, p, dangling_weights, is_dangling, alpha, max_iter=100, tol=1.0e-6):
    #print(M.shape)
    #print(x.shape)
    #print(p.shape)
    # power iteration: make up to max_iter iterations
    for i in range(max_iter):
        xlast = x
        x = alpha * (x * M + sum(x[is_dangling]) * dangling_weights) + \
            (1 - alpha) * p

        #x = alpha * (np.dot(x, M.todense()) + sum(x[is_dangling]) * dangling_weights) + \
        #    (1 - alpha) * p
        #print(x.shape)

        # check convergence, l1 norm
        x = x / x.sum()
        err = scipy.absolute(x - xlast).sum()
        if err < N * tol:
            #return dict(zip(nodelist, map(float, x)))
            #print(i)
            return x

    return x


def pagerank_fast(G, personal_vec):
    M = nx.to_scipy_sparse_matrix(G, nodelist=G.nodes(), weight='weight',
                                  dtype=float)
    S = scipy.array(M.sum(axis=1)).flatten()
    S[S != 0] = 1.0 / S[S != 0]
    Q = scipy.sparse.spdiags(S.T, 0, *M.shape, format='csr')
    M = Q * M

    ppr_mat = []
    for i in range(personal_vec.shape[1]):
        pr=pagerank_power(M, p=0.85, personalize=personal_vec[:, i], tol=1e-6)
        #pr=pagerank(M, p=0.85, personalize=personal_vec[:, i])
        ppr_mat.append(pr)
    return np.array(ppr_mat)

def pagerank_fast_mat(M, personal_vec):
    S = scipy.array(M.sum(axis=1)).flatten()
    S[S != 0] = 1.0 / S[S != 0]
    Q = scipy.sparse.spdiags(S.T, 0, *M.shape, format='csr')
    M = Q * M

    ppr_mat = []
    for i in range(25000):
        st = time.time()
        pr=pagerank_power(M, p=0.85, personalize=personal_vec[:, i], tol=1e-6)
        #pr=pagerank(M, p=0.85, personalize=personal_vec[:, i])
        ppr_mat.append(pr)
    return np.array(ppr_mat)

def pagerank_scikit(G):
    M = nx.to_scipy_sparse_matrix(G, nodelist=G.nodes(), weight='weight',
                                  dtype=float)
    S = scipy.array(M.sum(axis=1)).flatten()
    S[S != 0] = 1.0 / S[S != 0]
    Q = scipy.sparse.spdiags(S.T, 0, *M.shape, format='csr')
    M = Q * M
    pagerank = PageRank()

    ppr_mat = []
    for i in range(M.shape[0]):
        seeds = {i: 1}
        pr = pagerank.fit_transform(M, seeds)
        #print(pr.shape)
        #print(pr)

        ppr_mat.append(pr)
    return np.array(ppr_mat)


def pagerank_lu(G, personal_vec):
    M = nx.to_scipy_sparse_matrix(G, nodelist=G.nodes(), weight='weight',
                                  dtype=float)
    S = scipy.array(M.sum(axis=1)).flatten()
    S[S != 0] = 1.0 / S[S != 0]
    Q = scipy.sparse.spdiags(S.T, 0, *M.shape, format='csr')
    M = Q * M

    alpha = 0.85
    LU = linalg.splu(scipy.sparse.eye(M.shape[0]) - alpha * M)
    ppr_mat = []
    for i in range(personal_vec.shape[1]):
        pr = LU.solve(personal_vec[i])
        ppr_mat.append(pr)

    return np.array(ppr)


def pagerank_umf(G, personal_vec):
    M = nx.to_scipy_sparse_matrix(G, nodelist=G.nodes(), weight='weight',
                                  dtype=float)
    S = scipy.array(M.sum(axis=1)).flatten()
    S[S != 0] = 1.0 / S[S != 0]
    Q = scipy.sparse.spdiags(S.T, 0, *M.shape, format='csr')
    M = Q * M

    alpha = 0.85
    A = scipy.sparse.eye(M.shape[0]) - alpha * M
    LU = umfpack.splu(A)
    ppr_mat = []
    for i in range(personal_vec.shape[1]):
        pr = LU.solve(personal_vec[:, i])
        pr = linalg.spsolve(A, personal_vec[:, i])
        ppr_mat.append(pr)


def prac():
    A = np.array([[1, 2],
                [1, 1]])

    b = np.array([1, 2])
    LU = linalg.spilu(scipy.sparse.csr_matrix(A))
    x = LU.solve(b)
    print(x)




if __name__ == '__main__':

    #import sys
    #sys.exit()
    
    g = nx.star_graph(10000)
    
    #n_num = 30000
    n_num = len(g.nodes())
    #e_num = 15000
    #edges = np.random.randint(0, int(n_num), (int(e_num), 2), dtype=np.int)
    #weight = np.array([1 for i in range(e_num)])
    #A = scipy.sparse.csr_matrix(([1 for i in range(e_num)], (edges[:, 0], edges[:, 1])),
    #                            shape=(n_num, n_num))


    
    # personalized vecを作る
    personal_vec = []
    for i in range(n_num):
        val = np.zeros(n_num)
        val[i] = 1
        personal_vec.append(val[np.newaxis, :])
    personal_vec = np.concatenate(personal_vec, axis=0).transpose()
    
    #s = time.time()
    #ppr = pagerank_scipy(g, personal_vec)
    #print(time.time() - s)

    #s = time.time()
    #ppr = pagerank_scipy(g, personal_vec, cy=True)
    #print(time.time() - s)

    #s = time.time()
    #ppr = pagerank_fast(g, personal_vec)
    #print(time.time() - s)

    s = time.time()
    ppr = pagerank_scikit(g)
    print(time.time() - s)

    s = time.time()
    ppr = pagerank_lu(g, personal_vec)
    print(time.time() - s)

    s = time.time()
    ppr = pagerank_umf(g, personal_vec)
    print(time.time() - s)
    
    #M = nx.to_scipy_sparse_matrix(g, nodelist=g.nodes(), weight='weight',
                                  #dtype=float)
    #S = scipy.array(M.sum(axis=1)).flatten()
    #S[S != 0] = 1.0 / S[S != 0]
    #Q = scipy.sparse.spdiags(S.T, 0, *M.shape, format='csr')
    #M = Q * M
    #M = M.todense()
    #print(0.85 * np.dot(ppr[0], M) + (1 - 0.85) * personal_vec[:, 0])
    #print(ppr[0])
    

    #ppr = pagerank_fast_mat(A, personal_vec)