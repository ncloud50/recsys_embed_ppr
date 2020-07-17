import SLIM_model
import evaluate
import pandas as pd
import numpy as np
import pickle
import time

from importlib import reload
import optuna

# データロード
data_dir = '../data/bpr/'
user_item_train_df = pd.read_csv(data_dir + 'user_item_train.csv')
user_item_test_df = pd.read_csv(data_dir + 'user_item_test.csv')
user_list = []
item_list = []
with open(data_dir + 'user_list.txt', 'r') as f:
    for l in f:
        user_list.append(l.replace('\n', ''))
        
with open(data_dir + 'item_list.txt', 'r') as f:
    for l in f:
        item_list.append(l.replace('\n', ''))


# ハイパラ
# alpha, l1_ratio, lin_model

def load_params():
    return pickle.load(open('result/best_param.pickle', 'rb'))

def time_since(runtime):
    mi = int(runtime / 60)
    sec = runtime - mi * 60
    return (mi, sec)


def objective(trial):
    start = time.time()
    # define model and fit
    alpha = trial.suggest_loguniform('alpha', 1e-6, 1)
    l1_ratio = trial.suggest_uniform('l1_ratio', 0, 1)
    #lin_model = trial.suggest_categorical('lin_model', ['lasso', 'elastic'])
    
    model = SLIM_model.SLIM(alpha, l1_ratio, len(user_list), len(item_list), lin_model=lin_model)
    model.fit_glm(user_item_train_df)
    #model.fit(user_item_train_df)
    #model.fit_multi(user_item_train_df)
    #model.load_sim_mat('./sim_mat.txt', user_item_train_df)

    # evaluate
    eval_model = evaluate.Evaluater(user_item_test_df, len(user_list))
    score_sum = 0
    not_count = 0
    for i in range(len(user_list)):
        rec_item_idx = model.pred_ranking(i)
        score = eval_model.topn_precision(rec_item_idx, i)
        if score > 1:
            not_count += 1
            continue
        score_sum += score

        #if i > 20:
        #    break


    mi, sec = time_since(time.time() - start)
    print('{}m{}sec'.format(mi, sec))

    return -1 * (score_sum / (len(user_list) - not_count))


if __name__ == '__main__':
    params = load_params()
    alpha = params['alpha']
    l1_ratio = params['l1_ratio']

    lin_model = 'elastic'
    model = SLIM_model.SLIM(alpha, l1_ratio, len(user_list), len(item_list), lin_model=lin_model)
    model.fit_multi(user_item_train_df)
    eval_model = evaluate.Evaluater(user_item_test_df, len(user_list))
    score_sum = 0
    not_count = 0
    for i in range(len(user_list)):
        rec_item_idx = model.pred_ranking(i)
        score = eval_model.topn_precision(rec_item_idx, i)
        if score > 1:
            not_count += 1
            continue
        score_sum += score


    score = -1 * (score_sum / (len(user_list) - not_count))
    np.savetxt('score.txt', np.array([score]))

