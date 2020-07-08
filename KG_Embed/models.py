import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

device='cpu'

class DistMulti(nn.Module):

    def __init__(self, embedding_dim, relation_size, entity_size):
        super(DistMulti, self).__init__()
        self.embedding_dim = embedding_dim

        self.entity_embed = nn.Embedding(entity_size, embedding_dim)
        self.relation_embed = nn.Embedding(relation_size, embedding_dim)
        
        
    def forward(self, head, tail, relation):
        head_embed = self.entity_embed(head)
        tail_embed = self.entity_embed(tail)
        relation_embed = self.relation_embed(relation)
        
        score = torch.sum(head_embed * tail_embed * relation_embed, axis=1)
        score = torch.sigmoid(score)
        
        return score
    
    def predict(self, head, tail, relation):
        head_embed = self.entity_embed(head)
        tail_embed = self.entity_embed(tail)
        relation_embed = self.relation_embed(relation)
        
        score = torch.sum(head_embed * tail_embed * relation_embed, axis=1)
        score = torch.sigmoid(score)
        
        return score

class TransE(nn.Module):

    def __init__(self, embedding_dim, relation_size, entity_size, gamma=1):
        super(TransE, self).__init__()
        self.embedding_dim = embedding_dim

        self.entity_embed = nn.Embedding(entity_size, embedding_dim)
        self.relation_embed = nn.Embedding(relation_size, embedding_dim)
        # model para init(normalize)

        # margin para
        self.gamma = gamma
        
        
    def forward(self, head, tail, relation, n_head, n_tail, n_relation):

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
    
    def predict(self, head, tail, relation):
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
    
    
class SparseTransE(nn.Module):

    def __init__(self, embedding_dim, relation_size, entity_size, gamma=1, alpha=1e-4):
        super(SparseTransE, self).__init__()
        self.embedding_dim = embedding_dim

        self.entity_embed = nn.Embedding(entity_size, embedding_dim)
        self.relation_embed = nn.Embedding(relation_size, embedding_dim)
        # model para init(normalize)

        # margin para
        self.gamma = gamma
        
        # 正則化パラメータ
        self.alpha = alpha
        
        
    def forward(self, head, tail, relation, n_head, n_tail, n_relation, 
                reg_user, reg_item, reg_brand):

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
        reg_u = self.entity_embed(reg_user)
        reg_i = self.entity_embed(reg_item)
        if len(reg_brand) == 0:
            reg_b = torch.zeros(2, 2)
        else:
            reg_b = self.entity_embed(reg_brand)
        
        #print(reg_u.shape)
        #print(reg_i.shape)
        #print(reg_b.shape)
        reg = torch.norm(torch.mm(reg_u, reg_u.T)) + torch.norm(torch.mm(reg_i, reg_i.T)) \
            + torch.norm(torch.mm(reg_b, reg_b.T))

        score = score + self.alpha * reg
        
        return score
    
    def predict(self, head, tail, relation):
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
    