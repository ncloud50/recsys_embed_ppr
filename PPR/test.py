import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle
import time
import dataloader
import evaluate

import torch
import optuna



def load_params():
    return pickle.load(open('result/best_param.pickle', 'rb'))

def item_ppr(user, alpha):
    val = np.zeros(len(G.nodes))
    val[user] = 1
    k = [i for i in range(len(G.nodes))]
    personal_vec = dict(zip(k, val))
    #print(personal_vec)
    ppr = nx.pagerank_scipy(G, alpha=alpha)
    
    # random 後で消す
    #val = np.random.dirichlet([1 for i in range(len(G.nodes))], 1)[0]
    #k = [i for i in range(len(G.nodes))]
    #ppr = dict(zip(k, val))
    
    pred = []
    item_idx = [entity_list.index(i) for i in item_list]
    for i in item_idx:
        pred.append(ppr[i])
    
    return pred


def get_ranking_mat(alpha=0.85):
    ranking_mat = []
    count = 0
    for u in user_idx:
        pred = item_ppr(u, alpha)
        #print(pred)
        sorted_idx = np.argsort(np.array(pred))[::-1]
        ranking_mat.append(sorted_idx)
        
        #count += 1
        #if count > 100:
        #    break
            
    return ranking_mat


def topn_precision(ranking_mat, user_items_dict, n=10):
    not_count = 0
    precision_sum = 0
        
    for i in range(len(ranking_mat)):
        if len(user_items_dict[user_idx[i]]) == 0:
            not_count += 1
            continue
        sorted_idx = ranking_mat[i]
        topn_idx = sorted_idx[:n]  
        hit = len(set(topn_idx) & set(user_items_dict[user_idx[i]]))
        precision = hit / len(user_items_dict[user_idx[i]])
        precision_sum += precision
        
    return precision_sum / (len(user_idx) - not_count)


def time_since(runtime):
    mi = int(runtime / 60)
    sec = int(runtime - mi * 60)
    return (mi, sec)


if __name__ == '__main__':
    #params = load_params()
    #alpha = params['alpha']
    # データ読み込み

    data_path = '../data_luxury_5core/test'

    entity_list = []
    user_list =[]
    item_list = []
    with open(data_path + '/entity_list.txt', 'r') as f:
        for l in f:
            entity_list.append(l.replace('\n', ''))
            
    with open(data_path + '/user_list.txt', 'r') as f:
        for l in f:
            user_list.append(l.replace('\n', ''))
            
    with open(data_path + '/item_list.txt', 'r') as f:
        for l in f:
            item_list.append(l.replace('\n', ''))
            

    user_idx = [entity_list.index(u) for u in user_list]

    dataset = dataloader.AmazonDataset(data_path)
    # グラフを作る
    # user-itemとitem-userどちらの辺も追加
    edges = [[r[0], r[1]] for r in dataset.triplet_df.values]
    for r in dataset.triplet_df.values:
        if r[2] == 0:
            edges.append([r[1], r[0]])
        
    # load network
    G = nx.DiGraph()
    G.add_nodes_from([i for i in range(len(dataset.entity_list))])
    G.add_edges_from(edges)

    alpha = 0.85
    ranking_mat = get_ranking_mat(alpha)
    evaluater = evaluate.Evaluater(data_path)
    score = evaluater.topn_map(ranking_mat)
    np.savetxt('score.txt', np.array([score]))