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
from models import DistMulti, TransE
from training import TrainIterater
from evaluate import Evaluater

import optuna
import time 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import warnings
warnings.filterwarnings('ignore')


# dataload
model_name = 'TransE'
dataset = AmazonDataset('./data', model_name='TransE')
edges = [[r[0], r[1]] for r in dataset.triplet_df.values]


# load network
G = nx.DiGraph()
G.add_nodes_from([i for i in range(len(dataset.entity_list))])
G.add_edges_from(edges)


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
     
    print(u_i_mat.shape)
    print(i_i_b_mat.shape)
    print(i_i_v_mat.shape)
    print(i_b_mat.shape)
    u_i_mat = mat_to_graph(user_index, item_index, u_i_mat)
    i_i_b_mat = mat_to_graph(item_index, item_index, i_i_b_mat)
    i_i_v_mat = mat_to_graph(item_index, item_index, i_i_v_mat)
    i_b_mat = mat_to_graph(item_index, brand_index, i_b_mat)

    return u_i_mat + i_i_b_mat + i_i_v_mat + i_b_mat

            
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



if __name__ == '__main__':
    # model load
    #model = pickle.load(open('model.pickle', 'rb'))

    # とりあえず初期化したモデルで動かす
    embedding_dim = 16
    relation_size = len(set(list(dataset.triplet_df['relation'].values)))
    entity_size = len(dataset.entity_list)
    model = TransE(int(embedding_dim), relation_size, entity_size).to(device)

    re_kg = reconstruct_kg(model)
    print(re_kg.shape)
    print(type(re_kg))
