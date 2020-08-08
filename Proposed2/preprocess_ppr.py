import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle
import time

import torch
import optuna


# データ読み込み
data_dir = '../data_luxury_5core/valid1/'
triplet_df = pd.read_csv(data_dir + 'triplet.csv')
edges = [[r[0], r[1]] for r in triplet_df.values]

entity_list = []
user_list =[]
item_list = []
with open(data_dir + 'entity_list.txt', 'r') as f:
    for l in f:
        entity_list.append(l.replace('\n', ''))
        
with open(data_dir + 'user_list.txt', 'r') as f:
    for l in f:
        user_list.append(l.replace('\n', ''))
        
with open(data_dir + 'item_list.txt', 'r') as f:
    for l in f:
        item_list.append(l.replace('\n', ''))
        
        
user_items_test_dict = pickle.load(open(data_dir + 'user_items_test_dict.pickle', 'rb'))


# グラフを作る
G = nx.DiGraph()
G.add_nodes_from([i for i in range(len(entity_list))])
G.add_edges_from(edges)

def item_ppr(user, alpha):
    val = np.zeros(len(G.nodes))
    val[user] = 1
    k = [i for i in range(len(G.nodes))]
    personal_vec = dict(zip(k, val))
    ppr = nx.pagerank_scipy(G, alpha, personalization=personal_vec)
    
    return list(ppr.values())

if __name__ == '__main__':
    alpha = 0.06
    user_idx = [entity_list.index(u) for u in user_list]
    ppr_mat = []
    for u in user_idx:
        pred = item_ppr(u, alpha)
        ppr_mat.append(pred)

    ppr_mat = np.array(ppr_mat)
    np.savetxt('ppr_mat.txt', ppr_mat)