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
    
    def predict(self, user_tensor, item_tensor):
        return 0
    