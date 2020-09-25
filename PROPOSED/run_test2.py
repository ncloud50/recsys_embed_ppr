import networkx as nx
from networkx.exception import NetworkXError
from sknetwork.ranking import PageRank

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse
import pickle

import torch
import torch.nn.functional as F

from dataloader import AmazonDataset
import models
from models import DistMulti, TransE, SparseTransE
from training import TrainIterater
from evaluate import Evaluater

import optuna
import time 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import warnings
warnings.filterwarnings('ignore')




def load_params():
    #return pickle.load(open('result_luxury_2cross/best_param_TransE.pickle', 'rb'))
    return pickle.load(open('result_luxury/best_param_TransE.pickle', 'rb'))

def train_embed(data_dir, params, model_name):
    # ハイパラ読み込み
    embedding_dim = params['embedding_dim']
    batch_size = params['batch_size']
    lr = params['lr']
    weight_decay = params['weight_decay']
    warmup = params['warmup']
    #lr_decay_every = params['lr_decay_every']
    lr_decay_every = 2
    lr_decay_rate = params['lr_decay_rate']
    if model_name == 'SparseTransE':
        alpha = params['alpha']
    
    # dataload
    dataset = AmazonDataset(data_dir, model_name='TransE')
    relation_size = len(set(list(dataset.triplet_df['relation'].values)))
    entity_size = len(dataset.entity_list)
    if model_name == 'TransE':
        model = TransE(int(embedding_dim), relation_size, entity_size).to(device)
    elif model_name == 'SparseTransE':
        model = SparseTransE(int(embedding_dim), relation_size, entity_size, alpha=alpha).to(device)
    iterater = TrainIterater(batch_size=int(batch_size), data_dir=data_dir, model_name=model_name)
    #iterater.iterate_epoch(model, lr=lr, epoch=3000, weight_decay=weight_decay, warmup=warmup,
    #                       lr_decay_rate=lr_decay_rate, lr_decay_every=lr_decay_every, eval_every=1e+5)
    iterater.iterate_epoch(model, lr=lr, epoch=100, weight_decay=weight_decay, warmup=warmup,
                           lr_decay_rate=lr_decay_rate, lr_decay_every=lr_decay_every, eval_every=1e+5)
    return model



def mk_sparse_sim_mat(model, dataset, gamma):
    item_idx = torch.tensor([dataset.entity_list.index(i) for i in dataset.item_list], 
                        dtype=torch.long, device=device)

    user_idx = torch.tensor([dataset.entity_list.index(u) for u in dataset.user_list], 
                        dtype=torch.long, device=device)

    brand_idx = torch.tensor([dataset.entity_list.index(b) for b in dataset.brand_list], 
                        dtype=torch.long, device=device)

    
    # ここもっと上手く書きたい
    item_embed = model.entity_embed(item_idx)
    #item_embed = item_embed / torch.norm(item_embed, dim=1).view(item_embed.shape[0], -1)
    # 負の要素は0にする
    item_sim_mat = F.relu(torch.mm(item_embed, torch.t(item_embed)))

    user_embed = model.entity_embed(user_idx)
    #user_embed = user_embed / torch.norm(user_embed, dim=1).view(user_embed.shape[0], -1)
    # 負の要素は0にする
    user_sim_mat = F.relu(torch.mm(user_embed, torch.t(user_embed)))

    brand_embed = model.entity_embed(brand_idx)
    #brand_embed = brand_embed / torch.norm(brand_embed, dim=1).view(brand_embed.shape[0], -1)
    # 負の要素は0にする
    brand_sim_mat = F.relu(torch.mm(brand_embed, torch.t(brand_embed)))

    # 100/p(p=90)分位数で閾値を設定 
    thre = np.percentile(np.concatenate([np.ravel(item_sim_mat.to('cpu').detach().numpy().copy()), 
                                         np.ravel(user_sim_mat.to('cpu').detach().numpy().copy()),
                                         np.ravel(brand_sim_mat.to('cpu').detach().numpy().copy())]), 1)

    item_sim_mat = F.relu(item_sim_mat - thre)
    user_sim_mat = F.relu(user_sim_mat - thre)
    brand_sim_mat = F.relu(brand_sim_mat - thre)

    item_sim_mat = gamma[0] * scipy.sparse.csr_matrix(item_sim_mat.to('cpu').detach().numpy().copy())
    user_sim_mat = gamma[1] * scipy.sparse.csr_matrix(user_sim_mat.to('cpu').detach().numpy().copy())
    brand_sim_mat = gamma[2] * scipy.sparse.csr_matrix(brand_sim_mat.to('cpu').detach().numpy().copy())

    M = scipy.sparse.block_diag((item_sim_mat, user_sim_mat, brand_sim_mat))
    M_ = np.array(1 - M.sum(axis=1) / np.max(M.sum(axis=1)))
                                    
    M = M / np.max(M.sum(axis=1)) + scipy.sparse.diags(M_.transpose()[0])

    #data = M.data
    #thre = np.percentile(data, 99)
    #data = data[data > thre]
    #M.data = data
    print(M.shape)
    print(len(M.data))
    return M



def pagerank_scipy(G, sim_mat,  personal_vec=None, alpha=0.85, beta=0.01,
                   max_iter=500, tol=1.0e-6, weight='weight',
                   dangling=None):
    
    N = len(G)
    if N == 0:
        return {}

    nodelist = G.nodes()
    M = nx.to_scipy_sparse_matrix(G, nodelist=nodelist, weight=weight,
                                  dtype=float)
    S = scipy.array(M.sum(axis=1)).flatten()
    S[S != 0] = 1.0 / S[S != 0]
    Q = scipy.sparse.spdiags(S.T, 0, *M.shape, format='csr')
    M = Q * M

    # 遷移行列とsim_matを統合
    #sim_mat = mk_sparse_sim_mat(G, item_mat)
    M = beta * M + (1 - beta) * sim_mat
    
    # initial vector
    x = scipy.repeat(1.0 / N, N)

    
    # Personalization vector
    p = personal_vec
 
    dangling_weights = p
    is_dangling = scipy.where(S == 0)[0]


    #print(x.shape)
    #print(M.shape)
    #print(p.shape)
    
    ppr_mat = []
    for i in range(p.shape[1]):
        ppr = power_iterate(N, M, x, p[:, i], dangling_weights[:, i], is_dangling, 
                            alpha, max_iter, tol)
        ppr_mat.append(ppr)
        
        #if i > 100:
        #    print(np.array(ppr_mat).shape)
        #    break 
        
    return np.array(ppr_mat)
    

def power_iterate(N, M, x, p, dangling_weights, is_dangling, alpha, max_iter=500, tol=1.0e-6):
    #print(M.shape)
    #print(x.shape)
    #print(p.shape)
    # power iteration: make up to max_iter iterations
    for i in range(max_iter):
        xlast = x
        x = alpha * (x * M + sum(x[is_dangling]) * dangling_weights) + \
            (1 - alpha) * p
        # check convergence, l1 norm
        x = x / x.sum()
        err = scipy.absolute(x - xlast).sum()
        if err < N * tol:
            #return dict(zip(nodelist, map(float, x)))
            #print(i)
            return x
    # pagerankの収束ちゃんとやっとく
    print(x.sum())
    print(err)
    print(N * tol)
    #raise NetworkXError('pagerank_scipy: power iteration failed to converge '
                        #'in %d iterations.' % max_iter)
        
    #return dict(zip(nodelist, map(float, x)))
    return x


def pagerank_scikit(G, sim_mat, user_idx, alpha, beta):
    nodelist = G.nodes()
    M = nx.to_scipy_sparse_matrix(G, nodelist=nodelist, weight='weight',
                                  dtype=float)
    S = scipy.array(M.sum(axis=1)).flatten()
    S[S != 0] = 1.0 / S[S != 0]
    Q = scipy.sparse.spdiags(S.T, 0, *M.shape, format='csr')
    M = Q * M
    M = beta * M + (1 - beta) * sim_mat

    pagerank = PageRank(damping_factor=alpha)

    ppr_mat = []
    print_every = int(len(user_idx) / 3)
    s = time.time()
    for i in user_idx:
        seeds = {i: 1}
        pr = pagerank.fit_transform(M, seeds)
        ppr_mat.append(pr)
        if (i + 1) % print_every == 0:
            print('{}% {}sec'.format(i / len(user_idx) * 100,
                                    time.time() - s))

    return np.array(ppr_mat)
    

def item_ppr(G, dataset, sim_mat, alpha, beta):
    
    # personal_vecを作る(eneity_size * user_size)
    user_idx = [dataset.entity_list.index(u) for u in dataset.user_list]
    personal_vec = []
    for u in user_idx:
        val = np.zeros(len(G.nodes()))
        val[u] = 1
        personal_vec.append(val[np.newaxis, :])
    personal_vec = np.concatenate(personal_vec, axis=0).transpose()
    
    #ppr = pagerank_torch(G, sim_mat, personal_vec, alpha, beta)
    #ppr = pagerank_scipy(G, sim_mat, personal_vec, alpha, beta)
    ppr = pagerank_scikit(G, sim_mat, user_idx, alpha, beta)
    
    item_idx = [dataset.entity_list.index(i) for i in dataset.item_list]
    pred = ppr[:, item_idx]
    #print(pred.shape)
    return pred



def get_ranking_mat(G, dataset, model, gamma, alpha=0.85, beta=0.01):
    ranking_mat = []
    #sim_mat = reconstruct_kg(model)
    sim_mat = mk_sparse_sim_mat(model, dataset, gamma)
    pred = item_ppr(G, dataset, sim_mat, alpha, beta)
    #print(pred.shape)
    for i in range(len(dataset.user_list)):
        sorted_idx = np.argsort(np.array(pred[i]))[::-1]
        ranking_mat.append(sorted_idx)
        #break
    return ranking_mat


def topn_precision(ranking_mat, user_items_dict, n=10):
    not_count = 0
    precision_sum = 0
    user_idx = [dataset.entity_list.index(u) for u in dataset.user_list]
        
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
    # hyper parameter
    alpha = trial.suggest_uniform('alpha', 0, 0.5)
    beta = trial.suggest_uniform('beta', 0, 0.5)
    gamma1 = trial.suggest_uniform('gamma1', 0, 1)
    gamma2 = trial.suggest_uniform('gamma2', 0, 1)
    gamma3 = trial.suggest_uniform('gamma3', 0, 1)
    gamma = [gamma1, gamma2, gamma3]
    
    data_dir = ['../data_luxury_5core/valid1', '../data_luxury_5core/valid2']
    score_sum = 0
    for i in range(len(data_dir)):
        # dataload
        dataset = AmazonDataset(data_dir[i], model_name='TransE')

        # load network
        edges = [[r[0], r[1]] for r in dataset.triplet_df.values]
        # user-itemとitem-userどちらの辺も追加
        for r in dataset.triplet_df.values:
            if r[2] == 0:
                edges.append([r[1], r[0]])

        G = nx.DiGraph()
        G.add_nodes_from([i for i in range(len(dataset.entity_list))])
        G.add_edges_from(edges)

        ranking_mat = get_ranking_mat(G, dataset, model[i], gamma, alpha, beta)
        #score = topn_precision(ranking_mat, user_items_test_dict)
        evaluater = Evaluater(data_dir[i])
        score = evaluater.topn_map(ranking_mat)
        score_sum += score

    mi, sec = time_since(time.time() - start)
    print('{}m{}sec'.format(mi, sec))
    
    return -1 * score_sum / 2


def main():
    # kg_embedハイパラ
    kgembed_param = pickle.load(open('./kgembed_params/best_param_TransE.pickle', 'rb'))
    start = time.time()
    model1 = train_embed('../data_luxury_5core/valid1', kgembed_param, 'TransE')
    model2 = train_embed('../data_luxury_5core/valid2', kgembed_param, 'TransE')
    model = [model1, model2]
    mi, sec = time_since(time.time() - start)
    print(mi, sec)

    #model = pickle.load(open('model.pickle', 'rb'))

    study = optuna.create_study()
    study.optimize(objective, n_trials=50)
    df = study.trials_dataframe() # pandasのDataFrame形式
    df.to_csv('./result_luxury_2cross/hyparams_result_TransE_relu.csv')
    with open('./result_luxury_2cross/best_param_TransE.pickle', 'wb') as f:
        pickle.dump(study.best_params, f)

if __name__ == '__main__':
    start = time.time()

    data_dir = '../data_luxury_5core/test'

    # train kg embed
    #kgembed_param = pickle.load(open('./kgembed_params_luxury/best_param_TransE.pickle', 'rb'))
    kgembed_param = pickle.load(open('./kgembed_params_luxury/best_param_TransE.pickle', 'rb'))
    model = train_embed(data_dir, kgembed_param, 'TransE')

    # load param
    params = load_params()
    alpha = params['alpha']
    beta = params['beta']
    gamma1 = params['gamma1']
    gamma2 = params['gamma2']
    gamma3 = params['gamma3']
    gamma = [gamma1, gamma2, gamma3]


    # dataload
    dataset = AmazonDataset(data_dir, model_name='TransE')

    # load network
    edges = [[r[0], r[1]] for r in dataset.triplet_df.values]
    # user-itemとitem-userどちらの辺も追加
    for r in dataset.triplet_df.values:
        if r[2] == 0:
            edges.append([r[1], r[0]])

    G = nx.DiGraph()
    G.add_nodes_from([i for i in range(len(dataset.entity_list))])
    G.add_edges_from(edges)

    ranking_mat = get_ranking_mat(G, dataset, model, gamma, alpha, beta)
    evaluater = Evaluater(data_dir)
    score = evaluater.topn_map(ranking_mat)

    mi, sec = time_since(time.time() - start)
    print('{}m{}sec'.format(mi, sec))

    np.savetxt('score_transe3.txt', np.array([score]))