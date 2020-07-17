import networkx as nx
from networkx.exception import NetworkXError

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse
import pickle

import torch

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


# dataload
data_dir = '../data'
dataset = AmazonDataset(data_dir, model_name='SparseTransE')
edges = [[r[0], r[1]] for r in dataset.triplet_df.values]
user_items_test_dict = pickle.load(open(data_dir + '/user_items_test_dict.pickle', 'rb'))


# load network
G = nx.DiGraph()
G.add_nodes_from([i for i in range(len(dataset.entity_list))])
G.add_edges_from(edges)

def load_params():
    return pickle.load(open('result/best_param_gamma_SparseTransE.pickle', 'rb'))

def train_embed(params, model_name):
    
    # ハイパラ読み込み
    embedding_dim = params['embedding_dim']
    batch_size = params['batch_size']
    lr = params['lr']
    weight_decay = params['weight_decay']
    #warmup = params['warmup']
    #lr_decay_every = params['lr_decay_every']
    warmup = 350
    lr_decay_every = 2
    lr_decay_rate = params['lr_decay_rate']
    if model_name == 'SparseTransE':
        alpha = params['alpha']
    
    relation_size = len(set(list(dataset.triplet_df['relation'].values)))
    entity_size = len(dataset.entity_list)
    if model_name == 'TransE':
        model = TransE(int(embedding_dim), relation_size, entity_size).to(device)
    elif model_name == 'SparseTransE':
        model = SparseTransE(int(embedding_dim), relation_size, entity_size, alpha=alpha).to(device)
    iterater = TrainIterater(batch_size=int(batch_size), data_dir=data_dir, model_name=model_name)
    score =iterater.iterate_epoch(model, lr=lr, epoch=3000, weight_decay=weight_decay, warmup=warmup,
                           lr_decay_rate=lr_decay_rate, lr_decay_every=lr_decay_every, eval_every=1e+5)
    return model



def mk_sparse_sim_mat(model, gamma):
    item_idx = torch.tensor([dataset.entity_list.index(i) for i in dataset.item_list], 
                        dtype=torch.long, device=device)

    user_idx = torch.tensor([dataset.entity_list.index(u) for u in dataset.user_list], 
                        dtype=torch.long, device=device)

    brand_idx = torch.tensor([dataset.entity_list.index(b) for b in dataset.brand_list], 
                        dtype=torch.long, device=device)

    
    # ここもっと上手く書きたい
    item_embed = model.entity_embed(item_idx)
    #item_embed = item_embed / torch.norm(item_embed, dim=1).view(item_embed.shape[0], -1)
    item_sim_mat = torch.mm(item_embed, torch.t(item_embed))
    item_sim_mat = gamma[0] * scipy.sparse.csr_matrix(item_sim_mat.to('cpu').detach().numpy().copy())

    user_embed = model.entity_embed(user_idx)
    #user_embed = user_embed / torch.norm(user_embed, dim=1).view(user_embed.shape[0], -1)
    user_sim_mat = torch.mm(user_embed, torch.t(user_embed))
    user_sim_mat = gamma[1] * scipy.sparse.csr_matrix(user_sim_mat.to('cpu').detach().numpy().copy())

    brand_embed = model.entity_embed(brand_idx)
    #brand_embed = brand_embed / torch.norm(brand_embed, dim=1).view(brand_embed.shape[0], -1)
    brand_sim_mat = torch.mm(brand_embed, torch.t(brand_embed))
    brand_sim_mat = gamma[2] * scipy.sparse.csr_matrix(brand_sim_mat.to('cpu').detach().numpy().copy())

    M = scipy.sparse.block_diag((item_sim_mat, user_sim_mat, brand_sim_mat))
    M_ = np.array(1 - M.sum(axis=1) / np.max(M.sum(axis=1)))
                                    
    M = M / np.max(M.sum(axis=1)) + scipy.sparse.diags(M_.transpose()[0])
    #print(type(M))
    #print(M.shape)
    return M


def reconstruct_kg(model):
    with torch.no_grad():
        batch_size = int(len(dataset.item_list) / 2)
        item_index = [dataset.entity_list.index(item) for item in dataset.item_list]
        user_index = [dataset.entity_list.index(user) for user in dataset.user_list]
        brand_index = [dataset.entity_list.index(brand) for brand in dataset.brand_list]

        # user-itemの組に対して予測
        u_i_mat = []
        for i in user_index:
            #pred = torch.tensor([], device=device)
            u_i_vec = np.array([])
            for j in range(int(len(dataset.item_list) / batch_size) + 1):
                # modelにuser,itemを入力
                user_tensor = torch.tensor([i for k in range(batch_size)], dtype=torch.long, device=device)
                item_tensor = torch.tensor(item_index[j*batch_size : (j+1)*batch_size],
                                            dtype=torch.long, device=device)
                ### user ->(buy) itemはrelationが0であることに注意 ###
                relation_tensor = torch.tensor([0 for k in range(batch_size)], dtype=torch.long, device=device)
                
                if len(user_tensor) > len(item_tensor):
                    user_tensor = torch.tensor([i for k in range(len(item_tensor))],
                                            dtype=torch.long, device=device)
                    relation_tensor = torch.tensor([0 for k in range(len(item_tensor))],
                                                    dtype=torch.long, device=device)

                pred = np.array(model.predict(user_tensor, item_tensor, relation_tensor).cpu()) 
                u_i_vec = np.concatenate([u_i_vec, pred])
            u_i_mat.append(u_i_vec)
        u_i_mat = np.array(u_i_mat)

        # item-itemの組に対して予測
        # relationは also_buyとalso_viewの二つ
        i_i_b_mat = []
        i_i_v_mat = []
        for i in item_index:
            #pred = torch.tensor([], device=device)
            i_i_b_vec = np.array([])
            i_i_v_vec = np.array([])
            for j in range(int(len(dataset.item_list) / batch_size) + 1):
                # modelにuser,itemを入力
                h_item_tensor = torch.tensor([i for k in range(batch_size)], dtype=torch.long, device=device)
                t_item_tensor = torch.tensor(item_index[j*batch_size : (j+1)*batch_size],
                                            dtype=torch.long, device=device)
                ### item ->(also_buy) itemはrelationが2であることに注意 ###
                ### item ->(also_view) itemはrelationが3であることに注意 ###
                b_relation_tensor = torch.tensor([2 for k in range(batch_size)], dtype=torch.long, device=device)
                v_relation_tensor = torch.tensor([3 for k in range(batch_size)], dtype=torch.long, device=device)
                
                if len(h_item_tensor) > len(t_item_tensor):
                    h_item_tensor = torch.tensor([i for k in range(len(t_item_tensor))],
                                            dtype=torch.long, device=device)
                    b_relation_tensor = torch.tensor([2 for k in range(len(t_item_tensor))],
                                                    dtype=torch.long, device=device)
                    v_relation_tensor = torch.tensor([3 for k in range(len(t_item_tensor))],
                                                    dtype=torch.long, device=device)

                b_pred = np.array(model.predict(h_item_tensor, t_item_tensor, b_relation_tensor).cpu()) 
                v_pred = np.array(model.predict(h_item_tensor, t_item_tensor, v_relation_tensor).cpu()) 
                i_i_b_vec = np.concatenate([i_i_b_vec, b_pred])
                i_i_v_vec = np.concatenate([i_i_v_vec, v_pred])
            i_i_b_mat.append(i_i_b_vec)
            i_i_v_mat.append(i_i_v_vec)
        i_i_b_mat = np.array(i_i_b_mat)
        i_i_v_mat = np.array(i_i_v_mat)

        # item-brandの組に対して予測
        i_b_mat = []
        for i in item_index:
            #pred = torch.tensor([], device=device)
            i_b_vec = np.array([])
            for j in range(int(len(dataset.item_list) / batch_size) + 1):
                # modelにuser,itemを入力
                item_tensor = torch.tensor([i for k in range(batch_size)], dtype=torch.long, device=device)
                brand_tensor = torch.tensor(brand_index[j*batch_size : (j+1)*batch_size],
                                            dtype=torch.long, device=device)
                ### item ->(belong) brandはrelationが1であることに注意 ###
                relation_tensor = torch.tensor([1 for k in range(batch_size)], dtype=torch.long, device=device)
                
                if len(item_tensor) > len(brand_tensor):
                    item_tensor = torch.tensor([i for k in range(len(brand_tensor))],
                                            dtype=torch.long, device=device)
                    relation_tensor = torch.tensor([1 for k in range(len(brand_tensor))],
                                                    dtype=torch.long, device=device)

                if j * batch_size > len(brand_tensor): 
                    break
                pred = np.array(model.predict(item_tensor, brand_tensor, relation_tensor).cpu()) 
                i_b_vec = np.concatenate([i_b_vec, pred])
            i_b_mat.append(i_b_vec)
        i_b_mat = np.array(i_b_mat)
     
    #print(u_i_mat.shape)
    #print(i_i_b_mat.shape)
    #print(i_i_v_mat.shape)
    #print(i_b_mat.shape)
    u_i_mat = mat_to_graph(user_index, item_index, u_i_mat)
    i_i_b_mat = mat_to_graph(item_index, item_index, i_i_b_mat)
    i_i_v_mat = mat_to_graph(item_index, item_index, i_i_v_mat)
    i_b_mat = mat_to_graph(item_index, brand_index, i_b_mat)

    kg_mat = u_i_mat + i_i_b_mat + i_i_v_mat + i_b_mat
    col_sum = kg_mat.sum(axis=1)
    col_sum[col_sum ==  0] = 1
    kg_mat /= col_sum
    kg_mat = scipy.sparse.csr_matrix(kg_mat)

    return kg_mat


def mat_to_graph(row_idx, col_idx, mat):
    row_new = []
    col_new = []
    data = []
    for i in range(len(row_idx)):
        for j in range(len(col_idx)):
            row_new.append(row_idx[i])
            col_new.append(col_idx[j])
            data.append(mat[i, j])

    size = len(dataset.entity_list)
    return scipy.sparse.csr_matrix((data, (row_new, col_new)), shape=(size, size))
    


def pagerank_scipy(G, sim_mat,  personal_vec=None, alpha=0.85, beta=0.01,
                   max_iter=500, tol=1.0e-6, weight='weight',
                   dangling=None):
    
    #import scipy.sparse

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


def item_ppr(sim_mat, alpha, beta):
    
    # personal_vecを作る(eneity_size * user_size)
    user_idx = [dataset.entity_list.index(u) for u in dataset.user_list]
    personal_vec = []
    for u in user_idx:
        val = np.zeros(len(G.nodes()))
        val[u] = 1
        personal_vec.append(val[np.newaxis, :])
    personal_vec = np.concatenate(personal_vec, axis=0).transpose()
    
    #ppr = pagerank_torch(G, sim_mat, personal_vec, alpha, beta)
    ppr = pagerank_scipy(G, sim_mat, personal_vec, alpha, beta)
    
    item_idx = [dataset.entity_list.index(i) for i in dataset.item_list]
    pred = ppr[:, item_idx]
    print(pred.shape)
    return pred



def get_ranking_mat(model, gamma, alpha=0.85, beta=0.01):
    ranking_mat = []
    #sim_mat = reconstruct_kg(model)
    sim_mat = mk_sparse_sim_mat(model, gamma)
    pred = item_ppr(sim_mat, alpha, beta)
    #print(pred.shape)
    for i in range(len(dataset.user_list)):
        sorted_idx = np.argsort(np.array(pred[i]))[::-1]
        ranking_mat.append(sorted_idx)
        #break
    return ranking_mat


user_idx = [dataset.entity_list.index(u) for u in dataset.user_list]


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
    # train kg embed
    kgembed_param = pickle.load(open('./best_param_SparseTransE.pickle', 'rb'))
    model = train_embed(kgembed_param, 'SparseTransE')

    # load param
    params = load_params()
    alpha = params['alpha']
    beta = params['beta']
    gamma1 = params['gamma1']
    gamma2 = params['gamma2']
    gamma3 = params['gamma3']
    gamma = [gamma1, gamma2, gamma3]

    ranking_mat = get_ranking_mat(model, gamma, alpha, beta)
    score = topn_precision(ranking_mat, user_items_test_dict)

    np.savetxt('score_transe.txt', np.array([score]))

