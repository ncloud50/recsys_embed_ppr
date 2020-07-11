from dataloader import AmazonDataset
import models
from models import DistMulti, TransE, SparseTransE
from training import TrainIterater
from evaluate import Evaluater

import optuna
import numpy as np
import pickle
import time

import torch
from importlib import reload

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ハイパラ
# 
# embed_dim
# batch_size
# weight_decay, lr, warmup, lr_decay_every, lr_decay_rate
# kg embed model
#model_name = 'TransE'

model_name = 'SparseTransE'

def time_since(runtime):
    mi = int(runtime / 60)
    sec = runtime - mi * 60
    return (mi, sec)

def objective(trial):
    start = time.time()
    import gc
    gc.collect()

    # データ読み込み
    dataset = AmazonDataset('./data2', model_name='SparseTransE')
    
    relation_size = len(set(list(dataset.triplet_df['relation'].values)))
    entity_size = len(dataset.entity_list)
    embedding_dim = trial.suggest_discrete_uniform('embedding_dim', 16, 128, 16)
    alpha = trial.suggest_loguniform('weight_decay', 1e-6, 1e-2) #SparseTransEの時だけ
    model = SparseTransE(int(embedding_dim), relation_size, entity_size).to(device)
    
    batch_size = trial.suggest_int('batch_size', 128, 512, 128)
    iterater = TrainIterater(batch_size=int(batch_size), data_dir='./data2', model_name=model_name)
    
    lr= trial.suggest_loguniform('lr', 1e-4, 1e-2)
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-6, 1e-2)
    
    #warmup = trial.suggest_int('warmup', 100, 500)
    #warmup = trial.suggest_int('warmup', 1, 5)
    warmup = 350
    #lr_decay_every = trial.suggest_int('lr_decay_every', 1, 10)
    lr_decay_every = 2
    lr_decay_rate = trial.suggest_uniform('lr_decay_rate', 0.5, 1)
    
    score =iterater.iterate_epoch(model, lr=lr, epoch=3000, weight_decay=weight_decay, warmup=warmup,
                           lr_decay_rate=lr_decay_rate, lr_decay_every=lr_decay_every, eval_every=1e+5)
    
    torch.cuda.empty_cache()


    mi, sec = time_since(time.time() - start)
    print('{}m{}sec'.format(mi, sec))
    
    return -1 * score

if __name__ == '__main__':
    study = optuna.create_study()
    study.optimize(objective, n_trials=20)
    df = study.trials_dataframe() # pandasのDataFrame形式
    df.to_csv('./hyparams_result_SparseTransE.csv')
    with open('best_param_SparseTransE.pickle', 'wb') as f:
        pickle.dump(study.best_params, f)

