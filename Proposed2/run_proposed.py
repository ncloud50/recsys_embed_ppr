import networkx as nx
from networkx.exception import NetworkXError

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse
import pickle

import torch
import torch.functional as F

from dataloader import AmazonDataset
from kg_model import DistMulti, TransE, SparseTransE
from model import PPR_TransE
from training import TrainIterater
from inference import Inference

import optuna
import time 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def time_since(runtime):
    mi = int(runtime / 60)
    sec = int(runtime - mi * 60)
    return (mi, sec)


model_name = 'TransE'
def objective(trial):
    start = time.time()
    # pagerank para 
    mu = trial.suggest_uniform('mu', 0, 1)
    alpha = trial.suggest_uniform('beta', 0, 0.5)
    kappa1 = trial.suggest_uniform('kappa1', 0, 1)
    kappa2 = trial.suggest_uniform('kappa2', 0, 1)
    kappa3 = trial.suggest_uniform('kappa3', 0, 1)
    kappa = [kappa1, kappa2, kappa3]

    # model para
    embedding_dim = int(trial.suggest_discrete_uniform('embedding_dim', 16, 128, 16))
    #alpha = trial.suggest_loguniform('alpha', 1e-6, 1e-2) #SparseTransEの時だけ

    # training para
    lambda_ = trial.suggest_uniform('lambada_', 0, 1)
    batch_size = trial.suggest_int('batch_size', 256, 512, 128)
    lr= trial.suggest_loguniform('lr', 1e-4, 1e-2)
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-6, 1e-2)
    warmup = trial.suggest_int('warmup', 10, 100)
    #lr_decay_every = trial.suggest_int('lr_decay_every', 1, 10)
    lr_decay_every = 2
    lr_decay_rate = trial.suggest_uniform('lr_decay_rate', 0.5, 1)


    data_dir = ['../data_luxury_5core/valid1', '../data_luxury_5core/valid2']
    score_sum = 0
    for i in range(len(data_dir)):

        dataset = AmazonDataset(data_dir[i], model_name='TransE')
        relation_size = len(set(list(dataset.triplet_df['relation'].values)))
        entity_size = len(dataset.entity_list)

        ppr_transe = PPR_TransE(embedding_dim, relation_size, entity_size,
                                data_dir[i], alpha, mu, kappa).to(device)

        iterater = TrainIterater(batch_size=int(batch_size), data_dir=data_dir[i], model_name=model_name)

        iterater.iterate_epoch(ppr_transe, lr=lr, epoch=3000, weight_decay=weight_decay, 
                                lambda_=lambda_, warmup=warmup,
                                lr_decay_rate=lr_decay_rate,
                                lr_decay_every=lr_decay_every, eval_every=1e+5)


        # inference
        inf = Inference(data_dir[i])
        score = inf.get_score(ppr_transe, kappa, mu, alpha)
        score_sum += score



    mi, sec = time_since(time.time() - start)
    print('{}m{}sec'.format(mi, sec))
    
    return -1 * score_sum / 2


if __name__ == '__main__':
    study = optuna.create_study()
    study.optimize(objective, n_trials=50)
    df = study.trials_dataframe() # pandasのDataFrame形式
    df.to_csv('./hyparams_result_TransE.csv')
    with open('./best_param_TransE.pickle', 'wb') as f:
        pickle.dump(study.best_params, f)