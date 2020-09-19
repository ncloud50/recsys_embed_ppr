import networkx as nx
from networkx.exception import NetworkXError
from fast_pagerank import pagerank
from fast_pagerank import pagerank_power

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
import sys

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import warnings
warnings.filterwarnings('ignore')



def train_embed(data_dir, params, model_name):
    # ハイパラ読み込み
    embedding_dim = params['embedding_dim']
    batch_size = params['batch_size']
    lr = params['lr']
    weight_decay = params['weight_decay']
    #warmup = params['warmup']
    warmup = 350
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
    iterater.iterate_epoch(model, lr=lr, epoch=3000, weight_decay=weight_decay, warmup=warmup,
                           lr_decay_rate=lr_decay_rate, lr_decay_every=lr_decay_every, eval_every=1e+5,
                           early_stop=True)
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
    item_sim_mat = gamma[0] * scipy.sparse.csr_matrix(item_sim_mat.to('cpu').detach().numpy().copy())

    user_embed = model.entity_embed(user_idx)
    #user_embed = user_embed / torch.norm(user_embed, dim=1).view(user_embed.shape[0], -1)
    # 負の要素は0にする
    user_sim_mat = F.relu(torch.mm(user_embed, torch.t(user_embed)))
    user_sim_mat = gamma[1] * scipy.sparse.csr_matrix(user_sim_mat.to('cpu').detach().numpy().copy())

    brand_embed = model.entity_embed(brand_idx)
    #brand_embed = brand_embed / torch.norm(brand_embed, dim=1).view(brand_embed.shape[0], -1)
    # 負の要素は0にする
    brand_sim_mat = F.relu(torch.mm(brand_embed, torch.t(brand_embed)))
    brand_sim_mat = gamma[2] * scipy.sparse.csr_matrix(brand_sim_mat.to('cpu').detach().numpy().copy())

    M = scipy.sparse.block_diag((item_sim_mat, user_sim_mat, brand_sim_mat))
    M_ = np.array(1 - M.sum(axis=1) / np.max(M.sum(axis=1)))
                                    
    M = M / np.max(M.sum(axis=1)) + scipy.sparse.diags(M_.transpose()[0])
    #print(type(M))
    #print(M.shape)
    return M


def pagerank_fast(G, sim_mat, personal_vec, alpha, beta):
    nodelist = G.nodes()
    M = nx.to_scipy_sparse_matrix(G, nodelist=nodelist, weight='weight',
                                  dtype=float)
    S = scipy.array(M.sum(axis=1)).flatten()
    S[S != 0] = 1.0 / S[S != 0]
    Q = scipy.sparse.spdiags(S.T, 0, *M.shape, format='csr')
    M = Q * M

    # 遷移行列とsim_matを統合
    #sim_mat = mk_sparse_sim_mat(G, item_mat)
    M = beta * M + (1 - beta) * sim_mat

    ppr_mat = []
    for i in range(personal_vec.shape[1]):
        #pr = pagerank_power(M, p=alpha, personalize=personal_vec[:, i])
        pr = pagerank(M, p=alpha, personalize=personal_vec[:, i])
        ppr_mat.append(pr)

    return ppr_mat


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
    ppr = pagerank_fast(G, sim_mat, personal_vec, alpha, beta)
    
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


def time_since(runtime):
    mi = int(runtime / 60)
    sec = int(runtime - mi * 60)
    return (mi, sec)



if __name__ == '__main__':
    M = pickle.load(open('model_sim.mat', 'rb'))
    print(M.shape)

    #print(M[M != 0].shape)

    # 値の分布を調べる
    print(len(M.data))


