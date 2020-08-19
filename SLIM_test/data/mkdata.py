import pandas as pd
import numpy as np

triplet_df = pd.read_csv('./triplet.csv')

user_item_train_df = triplet_df[triplet_df['relation'] == 0]

user_item_train_df.to_csv('user_item_train.csv', index=False)