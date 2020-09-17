import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from SLIM import SLIM, SLIMatrix
import time
import evaluate
import optuna
import sys

def get_rating_mat(user_item_train_df, user_num, item_num):
    # rating_mat
    row = np.array([r[0] for r in user_item_train_df.values], dtype=int)
    col = np.array([r[1] for r in user_item_train_df.values], dtype=int)
    data = np.ones(len(user_item_train_df), dtype=int)
    rating_mat = csr_matrix((data, (row, col)), shape = (user_num, item_num))

    return rating_mat


def load_data(user_item_train_df, user_num, item_num):
    #read training data stored as triplets <user> <item> <rating>

    rating_mat = get_rating_mat(user_item_train_df, user_num,item_num)
    trainmat = SLIMatrix(rating_mat)

    return trainmat


def train(model, params, trainmat):
    model.train(params, trainmat)



def load_model_csr(user_num, item_num):
    row = []
    col = []
    data = []
    with open('./model.csr') as f:
        i = 0
        for l in f:
            if l == '\n':
                continue

            for k in range(len(l.split(' ')[1:])):
                if k % 2 == 0:
                    col.append(int(l.split(' ')[1:][k]))
                    row.append(i)
                else:
                    data.append(float(l.split(' ')[1:][k].replace('\n', '')))
                
    row = np.array(row, dtype=int)
    col = np.array(col, dtype=int)
    data = np.array(data, dtype=float)

    return csr_matrix((data, (row, col)), shape = (item_num, item_num))

                
def predict(user_item_train_df, user_num, item_num):
    sim_mat = load_model_csr(user_num, item_num)
    rating_mat = get_rating_mat(user_item_train_df, user_num,item_num)

    pred_mat = np.dot(rating_mat, sim_mat)
    #pred_mat = rating_mat * sim_mat)
    rec_mat = pred_mat - rating_mat


    return rec_mat
    

def predict2(model, user_item_train_df, user_num, item_num):
    trainmat = load_data(user_item_train_df, user_num, item_num)
    output = model.predict(trainmat, nrcmds=10, outfile='output.txt')
    return output


def objective(trial):
    data_dirs = ['../data_beauty_2core_es/valid1/bpr/', '../data_beauty_2core_es/valid2/bpr/']


    l1r = trial.suggest_loguniform('l1r', 1e-6, 1)
    l2r = trial.suggest_loguniform('l2r', 1e-6, 1)

    params = {
            'l1r':l1r, 
            'l2r':l2r
            }

    score_sum = 0
    for data_dir in data_dirs:
        user_list = []
        item_list = []
        with open(data_dir + 'user_list.txt', 'r') as f:
            for l in f:
                user_list.append(l.replace('\n', ''))
                
        with open(data_dir + 'item_list.txt', 'r') as f:
            for l in f:
                item_list.append(l.replace('\n', ''))

        
        user_item_train_df = pd.read_csv(data_dir + 'user_item_train.csv')
        user_item_test_df = pd.read_csv(data_dir + 'user_item_test.csv')
        eval_model = evaluate.Evaluater(user_item_test_df, len(user_list))

        # train slim
        trainmat = load_data(user_item_train_df, len(user_list), len(item_list))
        model = SLIM()
        train(model, params, trainmat)
        model.save_model(modelfname='model.csr', mapfname='map.csr') # filename to save the item map

        # predict
        rec_mat = predict(user_item_train_df, len(user_list), len(item_list))

        # eval
        start = time.time()
        map_sum = 0
        not_count = 0
        for i in range(rec_mat.shape[0]):
            rec_idx = np.argsort(rec_mat.getrow(i).toarray())[::-1]
            rec_idx = np.array(rec_idx)[0, :]

            score = eval_model.topn_map(rec_idx, i)
            
            if score > 1:
                not_count += 1
                continue
            map_sum += score

        score_sum += map_sum / (len(user_list) - not_count)


    return -1 * score_sum / 2


if __name__ == '__main__':
    study = optuna.create_study()
    study.optimize(objective, n_trials=30)

    save_path = 'result_beauty'

    df = study.trials_dataframe() # pandasのDataFrame形式
    df.to_csv(save_path + '/hyparams.csv')
    # save best params 
    with open(save_path + '/best_param.pickle', 'wb') as f:
        pickle.dump(study.best_params, f)