from dataloader import AmazonDataset
import models
from models import DistMulti, TransE, SparseTransE, Complex
from training import TrainIterater, EarlyStop
from evaluate import Evaluater

import optuna
import numpy as np
import pickle
import time

import torch
from importlib import reload

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



model_name = 'Complex'
def load_params():
    return pickle.load(open('./result_luxury/best_param.pickle', 'rb'))

def time_since(runtime):
    mi = int(runtime / 60)
    sec = runtime - mi * 60
    return (mi, sec)

if __name__ == '__main__':
    params = load_params()

    import gc
    gc.collect()

    # dataload
    data_dir = '../data_luxury_5core/valid1'
    dataset = AmazonDataset(data_dir, model_name=model_name)
    # hyper parameter 
    relation_size = len(set(list(dataset.triplet_df['relation'].values)))
    entity_size = len(dataset.entity_list)
    embedding_dim = params['embedding_dim']
    batch_size = params['batch_size']
    lr = params['lr']
    weight_decay = params['weight_decay']
    warmup = 350
    lr_decay_every = 2
    lr_decay_rate = params['lr_decay_rate']



    iterater = TrainIterater(batch_size=int(batch_size), data_dir=data_dir, model_name=model_name)
    #model = TransE(int(embedding_dim), relation_size, entity_size).to(device)
    model = Complex(int(embedding_dim), relation_size, entity_size).to(device)

    # early stop
    s = time.time()
    es = EarlyStop(data_dir, model_name, 3)
    print(time.time() - s)

    total_loss = es.iterate_valid_loss(model, batch_size=2)
    print(total_loss)
    

    # training and test
    #score =iterater.iterate_epoch(model, lr=lr, epoch=3000, weight_decay=weight_decay, warmup=warmup,
    #                       lr_decay_rate=lr_decay_rate, lr_decay_every=lr_decay_every, eval_every=1e+5)
    
    torch.cuda.empty_cache()