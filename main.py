import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split


from Preprocessing.Json_to_df import *

train_path = "./squad.json"
dev_path = "./squad_dev.json"

df_train = squad_json_to_dataframe_train(train_path)
df_dev =  squad_json_to_dataframe_dev(dev_path)

df_train = df_train.drop_duplicates(subset = ['question','answer_start','c_id']).reset_index(drop = True) #removing duplicates

from Preprocessing.dataframe_preprocessing import *

Missmatched_NaN_checker(df_train)
df_na_handler(df_train)
Missmatched_NaN_checker(df_dev)
df_na_handler(df_dev)

y_train = label_binarizer(df_train, 'answer_start')

df_train_1, df_val_1, y_train_1_hot, y_val_1_hot = train_test_split(df_train, y_train, test_size=0.33, random_state=12)

for col in ['question', 'context', 'text']:
    df_train = text_preprocessor(df_train, col)
    df_val = text_preprocessor(df_val, col)




#Fasttext embeddings gotten through wget https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.vec.gz

from Preprocessing.Fasttext_embs import *

filename = "cc.en.300.vec"

vocab, vecs = fasttext_reader(filename)


