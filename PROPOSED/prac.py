import networkx as nx
from networkx.exception import NetworkXError
from fast_pagerank import pagerank
from fast_pagerank import pagerank_power

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse
import pickle
import sys

import torch
import torch.nn.functional as F


def torch_test():
    M = torch.rand(2, 3, device='cuda:0')
    M2 = torch.rand(2, 3, device='cuda:0')

    V = M.to('cpu').detach().numpy().copy()
    V2 = M2.to('cpu').detach().numpy().copy()

    V3 = np.concatenate([np.ravel(V), np.ravel(V2)])
    thre = np.percentile(V3, 90)
    
    F.relu(M - thre)
    F.relu(M2 - thre)
    


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
    for p in range(90, 101, 1):
        print(len(data[data > np.percentile(data, p)]))
    #print(np.percentile(data, 99))
    #print(len(data[data > np.percentile(data, 99)]))

    sys.exit()

    data = M.data
    print(len(data))
    thre = np.percentile(M.data, 99)
    
    print(len(M.data))


    #data[data < thre] = 0
    #M.data = data
    #print(len(M.data))
    #print(M.shape)