import pandas as pd
import numpy as np
import time 

def func():
    # データ読み込み
    user_item_df = pd.read_csv('../All_Beauty/user_item.csv')
    item_list = list(set(list(user_item_df['asin'])))
    user_list = list(set(list(user_item_df['reviewerID'])))
    print('item size: {}'.format(len(item_list)))
    print('user size: {}'.format(len(user_list)))

    # user_itemをID化
    
    start = time.time()
    user_item_list = []
    count = 0
    print(len(user_item_df))
    for row in user_item_df.values:
        user = user_list.index(row[0])
        item = item_list.index(row[1])
        user_item_list.append([user, item])
        count += 1
        if count > 1000:
            break

    user_item_df = pd.DataFrame(np.array(user_item_list),
                                columns = ['reviewerID', 'asin'])

    # train-testスプリット
    user_item_df = user_item_df.take(np.random.permutation(len(user_item_df)))
    train_num = int(0.5 * len(user_item_df))
    user_item_train_df = user_item_df[0:train_num]
    user_item_test_df = user_item_df[train_num:]

    print('train {}'.format(train_num))
    print('test {}'.format(len(user_item_test_df)))
    # スプリットを保存
    #user_item_train_df.to_csv('./data/user_item_train.csv', index=False)
    #user_item_test_df.to_csv('./data/user_item_test.csv', index=False)

    runtime = time.time() - start
    print(runtime)

if __name__ == '__main__':
    func()
