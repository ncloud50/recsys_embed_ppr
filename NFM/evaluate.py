import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import dataloader

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Evaluater():


    def __init__(self, data_dir):
        self.dataset = dataloader.AmazonDataset(data_dir)

        
    def topn_precision(self, model, n=10):
        # user-itemの組に対して予測

        precision_sum = 0
        not_count = 0
        with torch.no_grad():

            batch_size = 2000
            item_index = [i for i in range(len(self.dataset.item_list))]
            for i in range(len(self.dataset.user_list)):
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

                    if len(user_tensor) > len(item_tensor):
                        user_tensor = torch.tensor([i for k in range(len(item_tensor))],
                                               dtype=torch.long, device=device)

                    pred = torch.cat([pred, model.predict(user_tensor, item_tensor)])

                # 予測をソート
                sorted_idx = np.argsort(np.array(pred.cpu()))[::-1]

                # topnにtarget userの推薦アイテムがいくつ含まれているか
                topn_idx = sorted_idx[:n]
                hit = len(set(topn_idx) & set(self.dataset.user_items_test_dict[i]))
                #precision = hit / len(self.dataset.user_items_test_dict[i])
                precision = hit / n
                precision_sum += precision

        return precision_sum / (len(self.dataset.user_list) - not_count)


    def topn_recall(self, model, n=10):
        return 0
