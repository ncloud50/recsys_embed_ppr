import numpy as np
import scipy
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dataloader import AmazonDataset

from kg_model import TransE, SparseTransE, Complex

device='cpu'


class PPR_TransE(TransE):

    def __init__(self, embedding_dim, relation_size, entity_size,
                    data_dir, kappa, gamma=1):
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


        # mk_sim_matの係数
        self.kappa = kappa

        
    def predict(self, head, tail, relation, n_head, n_tail, n_relation, 
                reg_user=None, reg_item=None, reg_brand=None):

        score = self.forward(head, tail, relation, n_head, n_tail, n_relation, 
                             reg_user, reg_item, reg_brand)
        M = self.mk_sparse_sim_mat(self.kappa)

        # ここでpagerankに相当する計算

        #return score

        
    def mk_sparse_sim_mat(self, kappa):

        # ここもっと上手く書きたい
        item_embed = self.entity_embed(self.item_idx)
        item_sim_mat = torch.mm(item_embed, torch.t(item_embed))
        item_sim_mat = kappa[0] * scipy.sparse.csr_matrix(item_sim_mat.to('cpu').detach().numpy().copy())

        user_embed = self.entity_embed(self.user_idx)
        user_sim_mat = torch.mm(user_embed, torch.t(user_embed))
        user_sim_mat = kappa[1] * scipy.sparse.csr_matrix(user_sim_mat.to('cpu').detach().numpy().copy())

        brand_embed = self.entity_embed(self.brand_idx)
        brand_sim_mat = torch.mm(brand_embed, torch.t(brand_embed))
        brand_sim_mat = kappa[2] * scipy.sparse.csr_matrix(brand_sim_mat.to('cpu').detach().numpy().copy())

        M = scipy.sparse.block_diag((item_sim_mat, user_sim_mat, brand_sim_mat))
        M_ = np.array(1 - M.sum(axis=1) / np.max(M.sum(axis=1)))
                                        
        M = M / np.max(M.sum(axis=1)) + scipy.sparse.diags(M_.transpose()[0])
        return M