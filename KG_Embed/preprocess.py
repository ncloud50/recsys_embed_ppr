import pandas as pd
import numpy as np
import pickle

def negative_sampling(triplet_df, user_list, item_list, 
                      brand_list, entity_list, entity_type, relation_type):
    # 比率を考える
    pos_triplet = [list(row) for row in triplet_df.values]
    nega_triplet = []

    relation_list = triplet_df['relation']
    dice = [len(relation_list[relation_list == i]) / len(relation_list) for i in range(len(relation_type))]
    sampling_relation_num = np.random.multinomial(len(triplet_df), dice) 
    for i in range(len(sampling_relation_num)):
        relation = i
        count = 0
        while count < sampling_relation_num[i]:
            h_entity = np.random.randint(len(entity_list))
            t_entity = np.random.randint(len(entity_list))
            if [h_entity, t_entity, relation] in pos_triplet:
                continue
            if [h_entity, t_entity, relation] in nega_triplet:
                continue

            nega_triplet.append([h_entity, t_entity, relation])
            count += 1

    nega_triplet_df = pd.DataFrame(nega_triplet, columns = ['h_entity', 't_entity', 'relation'])

    #保存
    nega_triplet_df.to_csv('./data/nega_triplet.csv', index=False)

    return nega_triplet_df
    
    
    
    
if __name__ == '__main__':
    # データ読み込み
    user_item_df = pd.read_csv('../user_item.csv')
    item_brand_df = pd.read_csv('../item_brand.csv')
    item_buy_item_df = pd.read_csv('../item_buy_item.csv')
    item_view_item_df = pd.read_csv('../item_view_item.csv')

    entity_type = ['user', 'item', 'brand']
    relation_type = ['u_buy_i', 'i_belong_b', 'i_also_buy_i', 'i_also_view_i']

    # 各entity_typeのリストを作る
    item_list = list(set(list(user_item_df['asin'])))
    user_list = list(set(list(user_item_df['reviewerID'])))
    brand_list = list(set(list(item_brand_df['brand'])))
    # nanを除く

    brand_list.pop(0)

    print('item {}'.format(len(item_list)))
    print('user {}'.format(len(user_list)))
    print('brand {}'.format(len(brand_list)))

    # 保存
    with open('./data/user_list.txt', 'w') as f:
        for user in user_list:
            f.write(user + '\n')
    with open('./data/item_list.txt', 'w') as f:
        for item in item_list:
            f.write(item + '\n')
    with open('./data/brand_list.txt', 'w') as f:
        for brand in brand_list:
            f.write(brand + '\n')

    # entityのリストを一つに連結する
    # このリストを使ってentityのidxを管理
    entity_list = item_list + user_list + brand_list
    print('entity size: {}'.format(len(entity_list)))

    # 保存
    with open('./data/entity_list.txt', 'w') as f:
        for entity in entity_list:
            f.write(entity + '\n')



    # テストデータとしてuser-itemインタラクションをスプリットする
    user_item_df = user_item_df.take(np.random.permutation(len(user_item_df)))
    train_num = int(0.5 * len(user_item_df))
    user_item_train_df = user_item_df[0:train_num]
    user_item_test_df = user_item_df[train_num:]

    print('train {}'.format(train_num))
    print('test {}'.format(len(user_item_test_df)))

    # user_item_test_dfをID化する
    user_item_test = []
    for row in user_item_test_df.values:
        user = entity_list.index(row[0])
        item = entity_list.index(row[1])
        user_item_test.append([user, item, relation_type.index('u_buy_i')])

    user_item_test_df = pd.DataFrame(user_item_test, columns = ['reviewerID', 'asin', 'relation'])

    #保存
    user_item_train_df.to_csv('./data/user_item_test.csv', index=False)



    # 一つのtriplet dataframeを作る
    # これが訓練データになる
    # e_1, e_2, relation　が行
    triplet_df = []
    for row in user_item_train_df.values:
        user = entity_list.index(row[0])
        item = entity_list.index(row[1])
        triplet_df.append([user, item, relation_type.index('u_buy_i')])

    for row in item_brand_df.values:
        if row[0] not in entity_list:
            continue
        if row[1] not in entity_list:
            continue
        item = entity_list.index(row[0])
        brand = entity_list.index(row[1])
        triplet_df.append([item, brand, relation_type.index('i_belong_b')])

    for row in item_buy_item_df.values:
        if row[0] not in entity_list:
            continue
        item_id = entity_list.index(row[0])
        if type(row[1]) != str:
            continue
        also_i = row[1][1:-1].split(',')
        if len(also_i) == 0:
            continue

        for a_i in also_i:
            #print(a_i)
            if a_i[1:-1] not in entity_list: continue
            also_item_id = entity_list.index(a_i[1:-1])
            triplet_df.append([item_id, also_item_id, relation_type.index('i_also_buy_i')])


    for row in item_view_item_df.values:
        if row[0] not in entity_list:
            continue
        item_id = entity_list.index(row[0])
        if type(row[1]) != str:
            continue
        also_i = row[1][1:-1].split(',')
        if len(also_i) == 0:
            continue

        for a_i in also_i:
            #print(a_i)
            if a_i[1:-1] not in entity_list: continue
            also_item_id = entity_list.index(a_i[1:-1])
            triplet_df.append([item_id, also_item_id, relation_type.index('i_also_view_i')])


    triplet_df = pd.DataFrame(triplet_df, columns=['h_entity', 't_entity', 'relation'])

    #保存
    triplet_df.to_csv('./data/triplet.csv', index=False)



    # negative sampling
    nega_triplet_df = negative_sampling(triplet_df, user_list, item_list, 
                      brand_list, entity_list, entity_type, relation_type)


    # trainデータに対するtargetを作る
    y_train = np.array([1 for i in range(len(triplet_df))] + [0 for i in range(len(nega_triplet_df))])
    #保存
    np.savetxt('./data/y_train.txt', y_train)


    # データに含まれるuser-item1, item2, item3, ...を返す
    # 辞書
    def user_aggregate_item(df):
        user_items_dict = {}
        #for user in user_list:
        for i in range(len(item_list), len(item_list) + len(user_list)):
            items_df = df[df['reviewerID'] == i]
            user_items_dict[i] = list(items_df['asin'])
        return user_items_dict

    user_items_test_dict = user_aggregate_item(user_item_test_df)

    with open('./data/user_items_test_dict.pickle', 'wb') as f:
        pickle.dump(user_items_test_dict, f)
