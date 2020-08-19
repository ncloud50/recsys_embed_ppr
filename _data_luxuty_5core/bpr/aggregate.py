import pandas as pd
import pickle

def user_aggregate_item(user_list, df):
    user_items_dict = {}
    #for user in user_list:
    for i in range(len(user_list)):
        items_df = df[df['reviewerID'] == i]
        user_items_dict[i] = list(items_df['asin'])
    return user_items_dict


if __name__ == '__main__':
    user_list = []
    with open('../user_list.txt', 'r') as f:
        for l in f:
            user_list.append(l.replace('\n', '')) 

    test_df = pd.read_csv('./user_item_test.csv')
    user_items_test_dict = user_aggregate_item(user_list, test_df)
    with open('./user_items_test_dict.pickle', 'wb') as f:
        pickle.dump(user_items_test_dict, f)

