
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time

import dataloader
import evaluate
import model
import training

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == '__main__':
    dataset = dataloader.AmazonDataset('./data')
    embedding_dim = 4
    user_size = len(dataset.user_list)
    item_size = len(dataset.item_list)
    layer_size = 2
    nfm = model.NFM(embedding_dim, user_size, item_size, layer_size)
    iterater = training.TrainIterater(batch_size=3)
    iterater.iterate_epoch(nfm, 0.001, 3, eval_every=1)


