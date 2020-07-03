import networkx as nx
from networkx.exception import NetworkXError

import pandas as pd

import numpy as np
import scipy.sparse
import pickle


class PageRank():
    
    
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.load_data()
        
    def load_data(self):
        slim_train = pd.read_csv('./data/user_item_train_slim.csv')
triplet_df = pd.read_csv('./data/triplet.csv')
edges = [[r[0], r[1]] for r in triplet_df.values]

user_list = []
item_list = []
entity_list = []
with open('./data/user_list.txt', 'r') as f:
    for l in f:
        user_list.append(l.replace('\n', ''))
with open('./data/item_list.txt', 'r') as f:
    for l in f:
        item_list.append(l.replace('\n', ''))
with open('./data/entity_list.txt', 'r') as f:
    for l in f:
        entity_list.append(l.replace('\n', ''))