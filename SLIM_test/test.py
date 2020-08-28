import SLIM_model
import evaluate
import pandas as pd
import numpy as np
import pickle
import time

from importlib import reload
import optuna



# ハイパラ
# alpha, l1_ratio, lin_model

def load_params():
    return pickle.load(open('result/best_param.pickle', 'rb'))

def time_since(runtime):
    mi = int(runtime / 60)
    sec = runtime - mi * 60
    return (mi, sec)


if __name__ == '__main__':
    # パラメータロード
    params = load_params()
    alpha = params['alpha']
    l1_ratio = params['l1_ratio']
    
    # データロード
    data_dir = '../data_luxury_5core/test/bpr/'
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


    lin_model = 'elastic'
    model = SLIM_model.SLIM(alpha, l1_ratio, len(user_list), len(item_list), lin_model=lin_model)
    model.fit_multi(user_item_train_df)
    #model.load_sim_mat('./sim_mat.txt', user_item_train_df)
    eval_model = evaluate.Evaluater(user_item_test_df, len(user_list))
    model.predict()
    score_sum = 0
    not_count = 0
    for i in range(len(user_list)):
        rec_item_idx = model.pred_ranking(i)
        #score = eval_model.topn_precision(rec_item_idx, i)
        score = eval_model.topn_map(rec_item_idx, i)
        if score > 1:
            not_count += 1
            continue
        score_sum += score


    score = -1 * (score_sum / (len(user_list) - not_count))
    print(score)
    #np.savetxt('score.txt', np.array([score]))


