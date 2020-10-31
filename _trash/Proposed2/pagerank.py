import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle
import time

import torch
import optuna


# データ読み込み
triplet_df = pd.read_csv('../data_luxury_5core/triplet.csv')
edges = [[r[0], r[1]] for r in triplet_df.values]

entity_list = []
user_list =[]
item_list = []
with open('../data_luxury_5core/entity_list.txt', 'r') as f:
    for l in f:
        entity_list.append(l.replace('\n', ''))
        
with open('../data_luxury_5core/user_list.txt', 'r') as f:
    for l in f:
        user_list.append(l.replace('\n', ''))
        
with open('../data_luxury_5core/item_list.txt', 'r') as f:
    for l in f:
        item_list.append(l.replace('\n', ''))
        
user_items_test_dict = pickle.load(open('../data_luxury_5core/user_items_test_dict.pickle', 'rb'))


# グラフを作る
G = nx.DiGraph()
G.add_nodes_from([i for i in range(len(entity_list))])
G.add_edges_from(edges)

# tripletに重複が存在する
print('edges: {}'.format(len(G.edges)))
print('nodes: {}'.format(len(G.nodes)))


def item_ppr(alpha):
    ppr_mat = []
    for i in range(len(entity_list)):
        val = np.zeros(len(G.nodes))
        val[i] = 1
        k = [i for i in range(len(G.nodes))]
        personal_vec = dict(zip(k, val))
        #print(personal_vec)
        ppr = nx.pagerank_scipy(G, alpha=alpha)
        ppr_mat.append(list(ppr.values()))

    return np.array(ppr_mat)


mat = item_ppr(0.030872309473542248)
np.savetxt('./ppr_mat.txt', mat)