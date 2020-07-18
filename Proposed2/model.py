import numpy as np
import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dataloader import AmazonDataset

device='cpu'


class PPR_Embed(nn.Module):

    def __init__(self, embedding_dim, relation_size, entity_size, kg_model_name, 
                    data_dir, kappa, gamma=1, alpha=1e-4):
        super(PPR_Embed, self).__init__()

        # dataloader
        self.dataset = AmazonDataset(data_dir)
        self.item_idx = torch.tensor([self.dataset.entity_list.index(i) for i in self.dataset.item_list], 
                            dtype=torch.long, device=device)

        self.user_idx = torch.tensor([self.dataset.entity_list.index(u) for u in self.dataset.user_list], 
                            dtype=torch.long, device=device)

        self.brand_idx = torch.tensor([self.dataset.entity_list.index(b) for b in self.dataset.brand_list], 
                            dtype=torch.long, device=device)

        
        
        self.embedding_dim = embedding_dim
        self.entity_embed = nn.Embedding(entity_size, embedding_dim)
        self.relation_embed = nn.Embedding(relation_size, embedding_dim)
        # model para init(normalize)

        # margin para
        self.gamma = gamma
        
        # 正則化パラメータ
        self.alpha = alpha

        # kgmodel
        self.kg_model_name = kg_model_name
        self.kg_model = self.{'TransE': self.TransE, 
                            'SparseTransE': self.SparseTransE}

        # mk_sim_matの係数
        self.kappa = kappa

        
    def forward(self, head, tail, relation, n_head, n_tail, n_relation, 
                reg_user=None, reg_item=None, reg_brand=None):

        score = self.kg_model[self.kg_model_name](head, tail, relation, n_head, n_tail, n_relation, 
                                                    reg_user, reg_item, reg_brand):
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


    def TransE(self, head, tail, relation, n_head, n_tail, n_relation):
                reg_user=None, reg_item=None, reg_brand=None):

        h = self.entity_embed(head)
        t = self.entity_embed(tail)
        r = self.relation_embed(relation)
        n_h = self.entity_embed(n_head)
        n_t = self.entity_embed(n_tail)
        n_r = self.relation_embed(n_relation)

        batch_size = h.shape[0]
        # normalize
        h = h / torch.norm(h, dim=1).view(batch_size, -1)
        t = t / torch.norm(t, dim=1).view(batch_size, -1)
        r = r / torch.norm(r, dim=1).view(batch_size, -1)
        n_h = n_h / torch.norm(n_h, dim=1).view(batch_size, -1)
        n_t = n_t / torch.norm(n_t, dim=1).view(batch_size, -1)
        n_r = n_r / torch.norm(n_r, dim=1).view(batch_size, -1)

        score = self.gamma + torch.norm((h + r - t), dim=1) - torch.norm((n_h + n_r - n_t), dim=1)
        
        return score

    
    def predict_TransE(self, head, tail, relation):
        h = self.entity_embed(head)
        t = self.entity_embed(tail)
        r = self.relation_embed(relation)

        #print(h)
        batch_size = h.shape[0]
        # normalize
        h /= torch.norm(h, dim=1).view(batch_size, -1)
        t /= torch.norm(t, dim=1).view(batch_size, -1)
        r /= torch.norm(r, dim=1).view(batch_size, -1)

        pred =  -1 * torch.norm((h + r - t), dim=1)

        return pred

        
    def SparseTransE(self, head, tail, relation, n_head, n_tail, n_relation, 
                reg_user=None, reg_item=None, reg_brand=None):
        
        h = self.entity_embed(head)
        t = self.entity_embed(tail)
        r = self.relation_embed(relation)
        n_h = self.entity_embed(n_head)
        n_t = self.entity_embed(n_tail)
        n_r = self.relation_embed(n_relation)

        batch_size = h.shape[0]
        # normalize
        h = h / torch.norm(h, dim=1).view(batch_size, -1)
        t = t / torch.norm(t, dim=1).view(batch_size, -1)
        r = r / torch.norm(r, dim=1).view(batch_size, -1)
        n_h = n_h / torch.norm(n_h, dim=1).view(batch_size, -1)
        n_t = n_t / torch.norm(n_t, dim=1).view(batch_size, -1)
        n_r = n_r / torch.norm(n_r, dim=1).view(batch_size, -1)

        score = self.gamma + torch.norm((h + r - t), dim=1) - torch.norm((n_h + n_r - n_t), dim=1)
        
        # 正則化
        if len(reg_user) == 0:
            reg_u = torch.zeros(2, 2)
        else:
            reg_u = self.entity_embed(reg_user)

        if len(reg_item) == 0:
            reg_i = torch.zeros(2, 2)
        else:
            reg_i = self.entity_embed(reg_item)

        if len(reg_brand) == 0:
            reg_b = torch.zeros(2, 2)
        else:
            reg_b = self.entity_embed(reg_brand)

        reg_u = reg_u / torch.norm(reg_u, dim=1).view(reg_u.shape[0], -1)
        reg_i = reg_i / torch.norm(reg_i, dim=1).view(reg_i.shape[0], -1)
        reg_b = reg_b / torch.norm(reg_b, dim=1).view(reg_b.shape[0], -1)
        
        reg = torch.norm(torch.mm(reg_u, reg_u.T)) + torch.norm(torch.mm(reg_i, reg_i.T)) \
            + torch.norm(torch.mm(reg_b, reg_b.T))

        score = score + self.alpha * reg
        
        return score
    
    def predict_SparseTransE(self, head, tail, relation):

        h = self.entity_embed(head)
        t = self.entity_embed(tail)
        r = self.relation_embed(relation)

        batch_size = h.shape[0]
        # normalize
        h /= torch.norm(h, dim=1).view(batch_size, -1)
        t /= torch.norm(t, dim=1).view(batch_size, -1)
        r /= torch.norm(r, dim=1).view(batch_size, -1)

        pred =  -1 * torch.norm((h + r - t), dim=1)

        return pred
    
