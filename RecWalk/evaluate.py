import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.linear_model import ElasticNet, Ridge, Lasso
from dataloader import AmazonDataset

class Evaluater():
    
    
    def __init__(self, data_dir):
        #self.user_num = user_num
        self.dataset = AmazonDataset(data_dir=data_dir) 
            
    
    def topn_precision(self, ranking_mat):
        user_idx = [self.dataset.entity_list.index(u) for u in self.dataset.user_list]
        not_count = 0
        precision_sum = 0
        for i in range(len(user_idx)):
            if len(self.dataset.user_items_test_dict[user_idx[i]]) == 0:
                not_count += 1
                continue

            precision = self.__topn_precision(ranking_mat[i], user_idx[i])
            precision_sum += precision
            
        return precision_sum / (len(self.dataset.user_list) - not_count)


    def __topn_precision(self, sorted_idx, target_user_id, n=10):

        #if len(self.user_items_dict[target_user_id]) == 0:
        
        topn_idx = sorted_idx[:n]   
        #print(topn_idx)
        #print(user_items_test_dict[target_user_id])
        hit = len(set(topn_idx) & set(self.user_items_dict[target_user_id]))
    
        #precision = hit / len(self.user_items_dict[target_user_id])
        precision = hit / n
        # precision_sum += precision
                
        return precision


    def topn_map(self, sorted_idx, user):
        map_sum = 0
        not_count = 0
        if len(self.user_items_dict[user]) == 0:
            return 2

        precision_sum = 0
        for j in self.user_items_dict[user]:
            n = list(sorted_idx).index(j) + 1
            precision = self.topn_precision(sorted_idx, user, n)
            precision_sum += precision
        
        map_sum += precision_sum / len(self.user_items_dict[user])

        return map_sum
    
    
    def topn_recall(n=10):
        return 0