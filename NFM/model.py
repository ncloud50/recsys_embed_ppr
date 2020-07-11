import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class NFM(nn.Module):

    def __init__(self, embedding_dim, user_size, item_size, layer_size):
        super(NFM, self).__init__()
        self.embedding_dim = embedding_dim

        self.user_embed = nn.Embedding(user_size, embedding_dim)
        self.item_embed = nn.Embedding(item_size, embedding_dim)

        self.layers = [nn.Linear(embedding_dim, embedding_dim).to(device) for i in range(layer_size)]
        self.layer_size = layer_size
        
        
    def forward(self, user_tensor, item_tensor):
        # user, itemをembed
        user_embed = self.user_embed(user_tensor)
        item_embed = self.item_embed(item_tensor)
        
        interaction_embed = user_embed * item_embed

        for i in range(self.layer_size):
            interaction_embed = F.relu(interaction_embed)
            interaction_embed = self.layers[i](interaction_embed)

        interaction_embed = F.relu(interaction_embed)
        prob = torch.sigmoid(torch.sum(interaction_embed, 1))
        
        return prob
    
    def predict(self, user_tensor, item_tensor):
        # user, itemをembed
        user_embed = self.user_embed(user_tensor)
        item_embed = self.item_embed(item_tensor)
        
        interaction_embed = user_embed * item_embed

        for i in range(self.layer_size):
            interaction_embed = F.relu(interaction_embed)
            interaction_embed = self.layers[i](interaction_embed)

        interaction_embed = F.relu(interaction_embed)
        prob = torch.sigmoid(torch.sum(interaction_embed, 1))
        
        return prob