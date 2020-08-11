import pickle
import time
import copy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from dataloader import AmazonDataset
from evaluate import Evaluater
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device: {}'.format(device))

class TrainIterater():


    def __init__(self, batch_size, data_dir, model_name='DistMulti'):
        #self.dataset = dataloader.AmazonDataset('./data')
        #self.dataset = AmazonDataset('./data', model_name=model_name)
        self.data_dir = data_dir
        self.dataset = AmazonDataset(self.data_dir, model_name=model_name)
        self.batch_size = batch_size
        self.model_name = model_name


    def train(self, batch, loss_func, optimizer, model, lambda_):
        optimizer.zero_grad()

        if self.model_name == 'DistMulti' or self.model_name == 'Complex':
            triplet, y_train = batch
            h_entity_tensor = torch.tensor(triplet[:, 0], dtype=torch.long, device=device)
            t_entity_tensor = torch.tensor(triplet[:, 1], dtype=torch.long, device=device)
            relation_tensor = torch.tensor(triplet[:, 2], dtype=torch.long, device=device)
            y_train = torch.tensor(y_train, dtype=torch.float, device=device)

            pred = model(h_entity_tensor, t_entity_tensor, relation_tensor)
            loss = loss_func(pred, y_train)

        elif self.model_name == 'TransE':
            posi_batch, nega_batch, ppr_vec, ppr_idx = batch
            h = torch.tensor(posi_batch[:, 0], dtype=torch.long, device=device)
            t = torch.tensor(posi_batch[:, 1], dtype=torch.long, device=device)
            r = torch.tensor(posi_batch[:, 2], dtype=torch.long, device=device)

            n_h = torch.tensor(nega_batch[:, 0], dtype=torch.long, device=device)
            n_t = torch.tensor(nega_batch[:, 1], dtype=torch.long, device=device)
            n_r = torch.tensor(nega_batch[:, 2], dtype=torch.long, device=device)

            score, vec = model.predict(h, t, r, n_h, n_t, n_r, ppr_vec, ppr_idx)
            loss = lambda_ * torch.sum(score)
            loss += (1 - lambda_) * torch.norm(torch.tensor(ppr_vec, device=device) - vec)

        elif self.model_name == 'SparseTransE':
            posi_batch, nega_batch, batch_user, batch_item, batch_brand = batch
            h = torch.tensor(posi_batch[:, 0], dtype=torch.long, device=device)
            t = torch.tensor(posi_batch[:, 1], dtype=torch.long, device=device)
            r = torch.tensor(posi_batch[:, 2], dtype=torch.long, device=device)

            n_h = torch.tensor(nega_batch[:, 0], dtype=torch.long, device=device)
            n_t = torch.tensor(nega_batch[:, 1], dtype=torch.long, device=device)
            n_r = torch.tensor(nega_batch[:, 2], dtype=torch.long, device=device)

            reg_user = torch.tensor(batch_user, dtype=torch.long, device=device)
            reg_item = torch.tensor(batch_item, dtype=torch.long, device=device)
            reg_brand = torch.tensor(batch_brand, dtype=torch.long, device=device)

            pred = model(h, t, r, n_h, n_t, n_r,
                         reg_user, reg_item, reg_brand)

            loss = torch.sum(pred)

        loss.backward()
        optimizer.step()

        return loss


    def iterate_train(self, model, lr=0.001, weight_decay=0, lambda_=0.5, print_every=2000, plot_every=50):
        # lambda_は埋め込み誤差とPPR誤差のバランス

        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        # optimizer = optim.SGD(model.parameters(), lr=lr)

        loss_func = nn.BCELoss()

        print_loss_total = 0
        plot_loss_list = []
        plot_loss_total = 0

        if self.model_name == 'DistMulti' or self.model_name == 'Complex':
            train_num = len(self.dataset.triplet_df) + len(self.dataset.nega_triplet_df)
        elif self.model_name == 'TransE' or self.model_name == 'SparseTransE':
            train_num = len(self.dataset.triplet_df)

        start_time = time.time()

        for i in range(int(train_num / self.batch_size) + 1):

            batch = self.dataset.get_batch(batch_size=self.batch_size)
            loss = self.train(batch, loss_func, optimizer, model, lambda_)
            print_loss_total += loss.detach()
            plot_loss_total += loss.detach()

            # print_everyごとに現在の平均のlossと、時間、dataset全体に対する進捗(%)を出力
            if (i+1) % print_every == 0:
                runtime = time.time() - start_time
                mi, sec = self.time_since(runtime)
                avg_loss = print_loss_total / print_every
                data_percent = int(i * self.batch_size / train_num * 100)
                print('train loss: {:e}    processed: {}({}%)    {}m{}sec'.format(
                    avg_loss, i*self.batch_size, data_percent, mi, sec))
                print_loss_total = 0

            # plot_everyごとplot用のlossをリストに記録しておく
            if (i+1) % plot_every == 0:
                avg_loss = plot_loss_total / plot_every
                plot_loss_list.append(avg_loss)
                plot_loss_total = 0

        return plot_loss_list


    def time_since(self, runtime):
        mi = int(runtime / 60)
        sec = int(runtime - mi * 60)
        return (mi, sec)


    def iterate_epoch(self, model, lr, epoch, weight_decay=0, lambda_=0.5,
                      warmup=0, lr_decay_rate=1, lr_decay_every=10, eval_every=5, early_stop=False):
        #eval_model = Evaluater(self.data_dir, model_name=self.model_name)
        es = EarlyStop(self.data_dir, self.model_name, patience=3)
        plot_loss_list = []
        plot_score_list = []

        for i in range(epoch):
            plot_loss_list.extend(self.iterate_train(model, lr=lr, weight_decay=weight_decay,
                                                     lambda_=lambda_,  print_every=10000))

            # early stop
            if early_stop:
                pre_model = es.early_stop(model)
                if pre_model:
                    print('Early Stop eposh: {}'.format(i+1))
                    return eval_model.topn_map(pre_model)

            # lrスケジューリング
            if i > warmup:
                if (i - warmup) % lr_decay_every == 0:
                    lr = lr * lr_decay_rate

            if (i+1) % eval_every == 0:
                #score = eval_model.topn_precision(model)
                #print('epoch: {}  precision: {}'.format(i, score))
                score = eval_model.topn_map(model)
                print('epoch: {}  map: {}'.format(i, score))
                plot_score_list.append(score)

        #self._plot(plot_loss_list)
        #self._plot(plot_score_list)

        #return eval_model.topn_precision(model)
        return eval_model.topn_map(model)



    def _plot(self, loss_list):
        # ここもっとちゃんと書く
        plt.plot(loss_list)
        plt.show()



class EarlyStop():

    def __init__(self, data_dir, model_name, patience):
        self.dataset = AmazonDataset(data_dir, model_name)
        self.patience = patience
        self.model_name = model_name

        self.user_item_nega_df = self.negative_sampling()

        y_test = [1 for i in range(len(self.dataset.user_item_test_df))] \
                   + [0 for i in range(len(self.user_item_nega_df))]
        self.y_test = np.array(y_test)
        


        self.loss_list = []
        self.model_list = []


    def negative_sampling(self):
        implicit_feed = [list(r) for r in self.dataset.user_item_test_df.values]
        user_idx = [self.dataset.entity_list.index(u) for u in self.dataset.user_list]
        item_idx = [self.dataset.entity_list.index(i) for i in self.dataset.item_list]

        user_item_test_nega = []
        count = 0
        while count < len(self.dataset.user_item_test_df):
            uidx = np.random.randint(len(self.dataset.user_list))
            iidx = np.random.randint(len(self.dataset.item_list))
            user = user_idx[uidx]
            item = item_idx[iidx]
            ### relationはすべてuser->(buy)itemの0
            if [user, item, 0] in implicit_feed:
                continue
            if [user, item, 0] in user_item_test_nega:
                continue

            user_item_test_nega.append([user, item, 0])
            count += 1

        user_item_test_nega_df = pd.DataFrame(user_item_test_nega, columns=['reviewerID', 'asin', 'relation'])

        return user_item_test_nega_df


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
        if self.model_name == 'DistMulti' or self.model_name == 'Complex':
            train_num = len(self.dataset.user_item_test_df) + len(self.user_item_nega_df)
            batch_idx = np.random.permutation(train_num)[:batch_size]
            # posi_tripletとnega_tripletを連結
            batch = pd.concat([self.dataset.user_item_test_df, self.user_item_nega_df]).values[batch_idx]
            batch_y_test = self.y_test[batch_idx]
        
            return batch, batch_y_test

        elif self.model_name == 'TransE':
            batch_idx = np.random.permutation(len(self.dataset.user_item_test_df))[:batch_size]
            posi_batch = self.dataset.user_item_test_df.values[batch_idx]
            nega_batch = self.user_item_nega_df.values[batch_idx]
            
            return posi_batch, nega_batch
            
        elif self.model_name == 'SparseTransE':
            batch_idx = np.random.permutation(len(self.dataset.user_item_test_df))[:batch_size]
            posi_batch = self.dataset.user_item_test_df.values[batch_idx]
            nega_batch = self.user_item_nega_df.values[batch_idx]

            # reguralizationのためのbatch
            # entity_typeの数だけ
            batch_entity_size = int(len(self.dataset.entity_list) / (len(self.dataset.user_item_test_df) / batch_size))
            reg_batch_idx = np.random.permutation(len(self.dataset.entity_list))[:batch_entity_size]

            batch_item = reg_batch_idx[reg_batch_idx < len(self.dataset.item_list)]

            batch_user = reg_batch_idx[reg_batch_idx >= len(self.dataset.item_list)]
            batch_user = batch_user[batch_user < len(self.dataset.user_list)]

            batch_brand = reg_batch_idx[reg_batch_idx >= len(self.dataset.user_list)]
            batch_brand = batch_brand[batch_brand < len(self.dataset.brand_list)]

            return posi_batch, nega_batch, batch_user, batch_item, batch_brand

        
    def iterate_valid_loss(self, model, batch_size=1024):
        loss_func = nn.BCELoss()
        loss_total = 0

        if self.model_name == 'DistMulti' or self.model_name == 'Complex':
            train_num = len(self.dataset.user_item_test_df) + len(self.user_item_nega_df)
        elif self.model_name == 'TransE' or self.model_name == 'SparseTransE':
            train_num = len(self.dataset.user_item_test_df)

        for i in range(int(train_num / batch_size) + 1):
            batch = self.get_batch(batch_size=batch_size)
            #print(batch)
            loss = self.valid_loss(batch, loss_func, model)
            #print(loss)
            loss_total += loss.detach()

        
        return loss_total / len(self.dataset.user_item_test_df)


    def valid_loss(self, batch, loss_func, model):

        with torch.no_grad(): 
            if self.model_name == 'DistMulti' or self.model_name == 'Complex':
                triplet, y_train = batch
                h_entity_tensor = torch.tensor(triplet[:, 0], dtype=torch.long, device=device)
                t_entity_tensor = torch.tensor(triplet[:, 1], dtype=torch.long, device=device)
                relation_tensor = torch.tensor(triplet[:, 2], dtype=torch.long, device=device)
                y_train = torch.tensor(y_train, dtype=torch.float, device=device)

                pred = model(h_entity_tensor, t_entity_tensor, relation_tensor)
                loss = loss_func(pred, y_train)

            elif self.model_name == 'TransE':
                posi_batch, nega_batch = batch
                h = torch.tensor(posi_batch[:, 0], dtype=torch.long, device=device)
                t = torch.tensor(posi_batch[:, 1], dtype=torch.long, device=device)
                r = torch.tensor(posi_batch[:, 2], dtype=torch.long, device=device)

                n_h = torch.tensor(nega_batch[:, 0], dtype=torch.long, device=device)
                n_t = torch.tensor(nega_batch[:, 1], dtype=torch.long, device=device)
                n_r = torch.tensor(nega_batch[:, 2], dtype=torch.long, device=device)

                pred = model(h, t, r, n_h, n_t, n_r)
                loss = torch.sum(pred)

            elif self.model_name == 'SparseTransE':
                posi_batch, nega_batch, batch_user, batch_item, batch_brand = batch
                h = torch.tensor(posi_batch[:, 0], dtype=torch.long, device=device)
                t = torch.tensor(posi_batch[:, 1], dtype=torch.long, device=device)
                r = torch.tensor(posi_batch[:, 2], dtype=torch.long, device=device)

                n_h = torch.tensor(nega_batch[:, 0], dtype=torch.long, device=device)
                n_t = torch.tensor(nega_batch[:, 1], dtype=torch.long, device=device)
                n_r = torch.tensor(nega_batch[:, 2], dtype=torch.long, device=device)

                reg_user = torch.tensor(batch_user, dtype=torch.long, device=device)
                reg_item = torch.tensor(batch_item, dtype=torch.long, device=device)
                reg_brand = torch.tensor(batch_brand, dtype=torch.long, device=device)

                pred = model(h, t, r, n_h, n_t, n_r,
                            reg_user, reg_item, reg_brand)

                loss = torch.sum(pred)
            
        return loss



    def valid_metric(self, model):
        return 0