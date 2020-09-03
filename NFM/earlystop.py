
import pandas as pd
import numpy as np
import time
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
        self.user_item_train_df = pd.read_csv(data_dir + '/user_item_train.csv')
        self.user_item_train_nega_df = pd.read_csv(data_dir + '/user_item_train_nega.csv')
        
        self.patience = patience

        y_train = [1 for i in range(len(self.user_item_train_df))] \
                   + [0 for i in range(len(self.user_item_train_nega_df))]
        self.y_train = np.array(y_train)
        
        self.loss_list = []
        self.model_list = []


    def early_stop(self, model):
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


    def get_batch(self, batch_size):
        train_num = len(self.user_item_train_df) + len(self.user_item_train_nega_df)
        batch_idx = np.random.permutation(train_num)[:batch_size]
        # posi_tripletとnega_tripletを連結
        batch = pd.concat([self.user_item_train_df, self.user_item_train_nega_df]).values[batch_idx]
        batch_y_train = self.y_train[batch_idx]
    
        return batch, batch_y_train

        
    def iterate_valid_loss(self, model, batch_size=1024):
        loss_func = nn.BCELoss()
        loss_total = 0

        train_num = len(self.user_item_train_df) + len(self.user_item_train_nega_df)

        for i in range(int(train_num / batch_size) + 1):
            batch = self.get_batch(batch_size=batch_size)
            loss = self.valid_loss(batch, loss_func, model)
            loss_total += loss.detach()
        
        return loss_total / len(self.user_item_train_df)


    def valid_loss(self, batch, loss_func, model):
        with torch.no_grad(): 
            batch, y_train = batch
            user_tensor = torch.tensor(batch[:, 0], dtype=torch.long, device=device)
            item_tensor = torch.tensor(batch[:, 1], dtype=torch.long, device=device)
            y_train = torch.tensor(y_train, dtype=torch.float, device=device)

            pred = model(user_tensor, item_tensor)
            loss = loss_func(pred, y_train)

        return loss


    def valid_metric(self, model):
        return 0

if __name__ == '__main__':
    import model
    dataset = AmazonDataset('../data_beauty_2core_es/valid1/bpr/')
    user_size = len(dataset.user_list)
    item_size = len(dataset.item_list)

    nfm = model.NFM(32, user_size, item_size, 2)
    es = EarlyStop('../data_beauty_2core_es/early_stopping/bpr', 10)
    es.early_stop(nfm)