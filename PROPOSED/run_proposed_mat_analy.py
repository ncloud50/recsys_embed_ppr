import networkx as nx
from networkx.exception import NetworkXError
from fast_pagerank import pagerank
from fast_pagerank import pagerank_power

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse
import pickle


if __name__ == '__main__':
    M = pickle.load(open('model_sim.mat', 'rb'))
    print(M.shape)
    

    #print(M[M != 0].shape)

    data = M.data
    print(data.shape)
    print(type(data))



    # 各統計量
    #print(pd.DataFrame(pd.Series(data.ravel()).describe()).transpose())

    # 100/p分位数
    #for p in range(0, 101, 10):
        #print(np.percentile(data, p))

    # 100/p分位数の数は？
    #for p in range(0, 101, 10):
    #    print(len(data[data > np.percentile(data, p)]))
    print(np.percentile(data, 99))
    print(len(data[data > np.percentile(data, 99)]))


    #data = M.data
    #print(len(data))
    #thre = np.percentile(M.data, 90)
    #data = data[data > thre]
    #M.data = data
    #print(len(M.data))