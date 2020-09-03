import pandas as pd
import numpy as np
import time
import pickle
from importlib import reload
import copy
import dataloader
import evaluate
from dataloader import AmazonDataset

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EarlyStop():

    def __init__(self, data_dir, patience):
        #self.dataset = AmazonDataset(data_dir)
        #self.user_item_nega_df = self.negative_sampling()

        # dataload
        self.user_item_train_df = pd.read_csv(data_dir + 'user_item_train.csv')
        self.user_item_train_nega_df = pd.read_csv(data_dir + 'user_item_train_nega.csv')
        self.user_items_nega_dict = pickle.load(open(data_dir + 'user_items_nega_dict.pickle', 'rb'))
        
        self.patience = patience

        #y_train = [1 for i in range(len(self.user_item_train_df))] \
        #           + [0 for i in range(len(self.user_item_train_nega_df))]
        #self.y_train = np.array(y_train)
        
        self.loss_list = []
        self.model_list = []


    def early_stop(self, model):
        self.item_size = model.item_embed.num_embeddings

        loss = self.iterate_valid_loss(model, batch_size=1024)
        self.loss_list.append(loss)
        # model copy
        self.model_list.append(copy.deepcopy(model))

        flag = 0
        for i in range(len(self.loss_list) - 1):
            if self.loss_list[0] > self.loss_list[i + 1]:
                flag = 1

        if flag == 0 and len(self.loss_list) > self.patience:
            return self.model_list[0]

        if len(self.loss_list) > self.patience:
            self.loss_list.pop(0)
            self.model_list.pop(0)

        return False


    def get_batch(self, batch_size=2):
        batch_idx = np.random.permutation(len(self.user_item_train_df))[:batch_size]
        batch = self.user_item_train_df.values[batch_idx]
        nega_batch = self.get_nega_batch(batch[:, 0])
    
        return batch, nega_batch
    
    
    def get_nega_batch(self, users):
        nega_batch = []
        for user in users:
            nega_items = self.user_items_nega_dict[user]
            #print(nega_items)
        
            # ここ直す
            if len(nega_items) == 0:
                #nega_batch.append([user, item_list[np.random.randint(item_num)]])
                nega_batch.append([user, np.random.randint(self.item_size)])
                continue
        
            nega_item = nega_items[np.random.randint(len(nega_items))]
            nega_batch.append([user, nega_item])
    
        return np.array(nega_batch)

        
    def iterate_valid_loss(self, model, batch_size=1024):
        loss_func = nn.BCELoss()
        loss_total = 0

        train_num = len(self.user_item_train_df) + len(self.user_item_train_nega_df)
        y_train = torch.ones(batch_size, dtype=torch.float, device=device)
        for i in range(int(train_num / batch_size) + 1):
            batch = self.get_batch(batch_size=batch_size)
            loss = self.valid_loss(batch, y_train, loss_func, model)
            loss_total += loss.detach()
        
        return loss_total / len(self.user_item_train_df)


    def valid_loss(self, batch, y_train, loss_func, model):
        with torch.no_grad(): 
            posi_batch, nega_batch = batch
            user_tensor = torch.tensor(posi_batch[:, 0], dtype=torch.long, device=device)
            item_tensor = torch.tensor(posi_batch[:, 1], dtype=torch.long, device=device)
            nega_item_tensor = torch.tensor(nega_batch[:, 1], dtype=torch.long, device=device)

            pred = model(user_tensor, item_tensor, nega_item_tensor)
            loss = loss_func(pred, y_train)

        return loss


    def valid_metric(self, model):
        return 0

if __name__ == '__main__':
    import bpr_model
    dataset = AmazonDataset('../data_beauty_2core_es/valid1/bpr/')
    user_size = len(dataset.user_list)
    item_size = len(dataset.item_list)

    model = bpr_model.BPR(32, user_size, item_size)

    es = EarlyStop('../data_beauty_2core_es/early_stopping/bpr/', 10)
    es.early_stop(model)