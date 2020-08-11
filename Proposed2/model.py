import numpy as np
import scipy
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dataloader import AmazonDataset

from kg_model import TransE, SparseTransE, Complex

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PPR_TransE(TransE):

    def __init__(self, embedding_dim, relation_size, entity_size,
                    data_dir, alpha, mu, kappa, gamma=1):
        super(PPR_TransE, self).__init__(embedding_dim, relation_size, entity_size, gamma)

        # dataloader
        self.dataset = AmazonDataset(data_dir)
        self.item_idx = torch.tensor([self.dataset.entity_list.index(i) for i in self.dataset.item_list], 
                            dtype=torch.long, device=device)

        self.user_idx = torch.tensor([self.dataset.entity_list.index(u) for u in self.dataset.user_list], 
                            dtype=torch.long, device=device)

        self.brand_idx = torch.tensor([self.dataset.entity_list.index(b) for b in self.dataset.brand_list], 
                            dtype=torch.long, device=device)

        # load network
        edges = [[r[0], r[1]] for r in self.dataset.triplet_df.values]
        # user-itemとitem-userどちらの辺も追加
        for r in self.dataset.triplet_df.values:
            if r[2] == 0:
                edges.append([r[1], r[0]])

        self.G = nx.DiGraph()
        self.G.add_nodes_from([i for i in range(len(self.dataset.entity_list))])
        self.G.add_edges_from(edges)
        self.H = nx.to_scipy_sparse_matrix(self.G)
        #self.H = scipy.sparse.coo_matrix(H)
        #coo = torch.tensor([H.row, H.col], dtype=torch.long)
        #v = torch.tensor(H.data, dtype=torch.float)
        #self.H = torch.sparse.FloatTensor(coo, v, torch.Size(H.shape), device=device)

        # mk_sim_matの係数
        self.kappa = kappa

        # 埋め込み誤差とページランク誤差のバランス
        # self.lambda_ = lambda_

        # 隣接行列と類似度行列のバランス
        self.alpha = alpha
        
        # PPRでのバイアスの強さ
        self.mu = mu
        

    def predict(self, head, tail, relation, n_head, n_tail, n_relation, 
                ppr_vec, ppr_user_idx):

        ppr_tensor = torch.tensor(ppr_vec, dtype=torch.float, device=device) 
        score = self.forward(head, tail, relation, n_head, n_tail, n_relation)

        # ここでpagerankに相当する計算
        M = self.mk_sparse_sim_mat()
        vec = torch.tensor([[] for i in range(ppr_tensor.shape[0])])
        pre_size = 0
        for k, mat in zip(self.kappa, M):
            size = mat.shape[0]
            tmp = (1 - self.alpha) * k * torch.mm(ppr_tensor[:, pre_size:size+pre_size], mat)
            vec = torch.cat([vec, tmp], dim=1)
            pre_size = size
            
        bias = []
        for i in ppr_user_idx:
            tmp = np.array([0 for j in range(len(self.dataset.entity_list))])
            tmp[i] = 1
            bias.append(tmp[np.newaxis])
        bias = np.concatenate(bias)
        
        # scipy.sparse matrixを使った計算
        vec_sparse = self.mu * self.alpha * ppr_vec * self.H + (1 - self.mu) * bias
        vec = torch.tensor(vec_sparse, device=device) + vec
        return score, vec

        
    def mk_sparse_sim_mat(self):

        # ここもっと上手く書きたい
        item_embed = self.entity_embed(self.item_idx)
        item_sim_mat = F.relu(torch.mm(item_embed, torch.t(item_embed)))
        #item_sim_mat = self.kappa[0] * scipy.sparse.coo_matrix(item_sim_mat.to('cpu').detach().numpy().copy())

        user_embed = self.entity_embed(self.user_idx)
        user_sim_mat = F.relu(torch.mm(user_embed, torch.t(user_embed)))
        #user_sim_mat = self.kappa[1] * scipy.sparse.coo_matrix(user_sim_mat.to('cpu').detach().numpy().copy())

        brand_embed = self.entity_embed(self.brand_idx)
        brand_sim_mat = F.relu(torch.mm(brand_embed, torch.t(brand_embed)))
        #brand_sim_mat = self.kappa[2] * scipy.sparse.coo_matrix(brand_sim_mat.to('cpu').detach().numpy().copy())

        def normalize(M):
            M_ = 1 - torch.sum(M, dim=1) / torch.max(torch.sum(M, dim=1))
            M = M / torch.max(torch.sum(M, dim=1)) + torch.diag(M_.T)
            return M

        item_sim_mat = normalize(item_sim_mat)
        user_sim_mat = normalize(user_sim_mat)
        brand_sim_mat = normalize(brand_sim_mat)

        return item_sim_mat, user_sim_mat, brand_sim_mat