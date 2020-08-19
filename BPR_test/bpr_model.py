import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class BPR(nn.Module):

    def __init__(self, embedding_dim, user_size, item_size):
        super(BPR, self).__init__()
        self.embedding_dim = embedding_dim

        self.user_embed = nn.Embedding(user_size, embedding_dim)
        self.item_embed = nn.Embedding(item_size, embedding_dim)
        
        
    def forward(self, user_tensor, item_tensor, nega_item_tensor):
        # user, itemをembed
        user_embed = self.user_embed(user_tensor)
        item_embed = self.item_embed(item_tensor)
        nega_item_embed = self.item_embed(nega_item_tensor)
        
        
        interaction_embed = torch.sum(user_embed * item_embed, 1)
        nega_interaction_embed = torch.sum(user_embed * nega_item_embed, 1)
        
        prob = torch.sigmoid(interaction_embed - nega_interaction_embed)
        
        return prob
    
    def predict(self, user_tensor, item_tensor):
        user_embed = self.user_embed(user_tensor)
        item_embed = self.item_embed(item_tensor)
        interaction_embed = torch.sum(user_embed * item_embed, 1)
        
        mu = torch.mean(interaction_embed)
        prob = torch.sigmoid(interaction_embed)
        
        return prob
    