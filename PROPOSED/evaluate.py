import pickle
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from dataloader import AmazonDataset
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Evaluater():


    def __init__(self, data_dir, model_name='DistMulti'):
        self.dataset = AmazonDataset(data_dir, model_name=model_name)
        self.model_name = model_name

        
    def topn_precision(self, model, n=10):
        # user-itemの組に対して予測

        precision_sum = 0
        not_count = 0
        with torch.no_grad():

            batch_size = int(len(self.dataset.item_list) / 2)
            item_index = [self.dataset.entity_list.index(item) for item in self.dataset.item_list]
            user_index = [self.dataset.entity_list.index(user) for user in self.dataset.user_list]
            for i in user_index:
                if len(self.dataset.user_items_test_dict[i]) == 0:
                    not_count += 1
                    continue

                pred = torch.tensor([], device=device)
                for j in range(int(len(self.dataset.item_list) / batch_size) + 1):
                    # modelにuser,itemを入力
                    # batchでやると速い
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

                    pred = torch.cat([pred, model.predict(user_tensor, item_tensor, relation_tensor)])

                # 予測をソート
                ### item_idxは0~len(item_list)-1 なのでこれでOK
                ### item_idxがentity_listの途中から始まっている場合は別
                sorted_idx = np.argsort(np.array(pred.cpu()))[::-1]

                topn_idx = sorted_idx[:n]
                hit = len(set(topn_idx) & set(self.dataset.user_items_test_dict[i]))
                #precision = hit / len(self.dataset.user_items_test_dict[i])
                precision = hit / n
                precision_sum += precision

        return precision_sum / (len(self.dataset.user_list) - not_count)


    def topn_recall(self, model, n=10):
        return 0