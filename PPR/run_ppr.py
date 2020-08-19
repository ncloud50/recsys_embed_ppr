import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle
import time

import torch
import optuna

import evaluate
import dataloader



def item_ppr(G, user, alpha, dataset):
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
    for i in dataset.item_idx:
        pred.append(ppr[i])
    
    return pred


def get_ranking_mat(G, alpha=0.85, dataset=None):
    ranking_mat = []
    count = 0
    for u in dataset.user_idx:
        pred = item_ppr(G, u, alpha, dataset)
        #print(pred)
        sorted_idx = np.argsort(np.array(pred))[::-1]
        ranking_mat.append(sorted_idx)
        
        #count += 1
        #if count > 2:
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
        #precision = hit / len(user_items_dict[user_idx[i]])
        precision = hit / n 
        precision_sum += precision
        
    return precision_sum / (len(user_idx) - not_count)


def time_since(runtime):
    mi = int(runtime / 60)
    sec = int(runtime - mi * 60)
    return (mi, sec)


def objective(trial):
    start = time.time()

    data_dir = ['../data_luxury_5core/valid1/', '../data_luxury_5core/valid2/']
    score_sum = 0

    for data_path in data_dir:
        # dataload
        dataset = dataloader.AmazonDataset(data_path)
        edges = [[r[0], r[1]] for r in dataset.triplet_df.values]
        for r in dataset.triplet_df.values:
            if r[2] == 0:
                edges.append([r[1], r[0]])

        #user_items_test_dict = pickle.load(open(data_path + 'user_items_test_dict.pickle', 'rb'))

        # グラフを作る
        G = nx.DiGraph()
        G.add_nodes_from([i for i in range(len(dataset.entity_list))])
        G.add_edges_from(edges)

        # ハイパラ
        alpha = trial.suggest_uniform('alpha', 0, 1)

        ranking_mat = get_ranking_mat(G, alpha, dataset)
        evaluater = evaluate.Evaluater(data_path)
        score = evaluater.topn_map(ranking_mat)
        score_sum += score


    mi, sec = time_since(time.time() - start)
    print('{}m{}s'.format(mi, sec))
    return -1 * score_sum / 2

if __name__ == '__main__':
    study = optuna.create_study()
    study.optimize(objective, n_trials=10)
    df = study.trials_dataframe() # pandasのDataFrame形式
    df.to_csv('hyparams_result_luxury.csv')