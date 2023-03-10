import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

#Downloading and basic preprocessing of the dataframe

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


#Downloading Fasttext embeddings and setting them up

#Fasttext embeddings gotten through wget https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.vec.gz

from Preprocessing.Fasttext_embs import *

filename = "cc.en.300.vec"
vocab, vecs = fasttext_reader(filename)

max_words = 100000
max_seq = 2000

train_data, val_data, train_data_q, val_data_q, word_index = tokenizer_preprocessing(df_train, df_val, max_words, max_seq)
embedding_matrix = embedding_matrix_maker(word_index, vocab, vecs, max_words)

#Running Model with Fasttext

from Models.Fasttext_based_Coattention import *

model = create_fasttext_coattention_model(embedding_matrix)

history = model.fit([train_data, train_data_q], 
        y_train_1_hot, batch_size = 128,
        epochs=5, validation_data=([val_data, val_data_q], y_val_1_hot))

#Running Model with BERT sentence embeddings

from Models.BERT_based_Coattention import *

preprocess_path = "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3"
BERT_path = "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-6_H-128_A-2/2"

model_w_BERT = create_BERT_coattention_model(BERT_path, preprocess_path, n_classes=5002, lr = 1e-3)

history_w_BERT = model_w_BERT.fit([df_train_1.context, df_train_1.question], 
        df_train_1.answer_start, batch_size = 128,
        epochs=5, validation_data=([df_val_1.context, df_val_1.question], df_val_1.answer_start))


#Running model after smoothing imbalance and binning text
from Preprocessing.imbalance_smoothing import *

df_train_1 = undersampling(df_train_1, 'answer_start')

df_train_1 = text_binning(df_train_1, 'answer_start', 'answer_bin', 10)
df_val_1 = text_binning(df_val_1, 'answer_start', 'answer_bin', 10)

smooth_model_w_BERT = create_BERT_coattention_model(BERT_path, preprocess_path, n_classes=501, lr = 1e-5)

history_smooth = smooth_model_w_BERT.fit([df_train_1.context, df_train_1.question], 
        df_train_1.answer_bin, batch_size = 128,
        epochs=5, validation_data=([df_val_1.context, df_val_1.question], df_val_1.bin))
