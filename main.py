import sys
from argparse import ArgumentParser

import sklearn
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from Preprocessing.json_to_df import *
from Preprocessing.fasttext_embs import *
from Preprocessing.dataframe_preprocessing import *
from Preprocessing.imbalance_smoothing import *

from Models.fasttext_based_coattention import *
from Models.bert_based_coattention import *



def prepare(train_path, dev_path):

    df_train = squad_json_to_dataframe_train(train_path)
    df_dev =  squad_json_to_dataframe_dev(dev_path)

    df_train = df_train.drop_duplicates(subset=['question', 'answer_start', 'c_id']).reset_index(drop=True) #removing duplicates

    Missmatched_NaN_checker(df_train)
    df_na_handler(df_train) 
    Missmatched_NaN_checker(df_dev)
    df_na_handler(df_dev)

    for col in ['question', 'context', 'text']:
        df_train = text_preprocessor(df_train, col)
        df_dev = text_preprocessor(df_dev, col)

    return df_train, df_dev




def train(df_train, model_choice, model_save_path):

    y_train = label_binarizer(df_train, 'answer_start')

    df_train_1, df_val_1, y_train_1_hot, y_val_1_hot = train_test_split(df_train, y_train, test_size=0.33, random_state=12)

    MAX_WORDS = 100000
    MAX_SEQ = 2000
    BATCH_SIZE = 128
    EPOCHS = 5

    if model_choice == 'fasttext': 

        df_train_1 = df_train_1[:100]
        df_val_1 = df_val_1[:40]

    
        #Fasttext embeddings gotten through wget https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.vec.gz
        embeddings_filename = "cc.en.300.vec"
        vocab, vecs = fasttext_reader(embeddings_filename)

        #Preprocessing data for use with the embedding layer:

        train_data, val_data, train_data_q, val_data_q, word_index = tokenizer_preprocessing(df_train_1, df_val_1, MAX_WORDS, MAX_SEQ)

        embedding_matrix = embedding_matrix_maker(word_index, vocab, vecs, MAX_WORDS)

        #Running Model with Fasttext
        fasttext_model = create_fasttext_coattention_model(embedding_matrix)

        fasttext_model.fit([train_data, train_data_q],
                y_train_1_hot, batch_size=BATCH_SIZE,
                epochs=EPOCHS, validation_data=([val_data, val_data_q], y_val_1_hot))
        
        fasttext_model.save(model_save_path)

    elif model_choice == 'big_bert':

        #Running Model with BERT sentence embeddings
        preprocess_path = "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3"
        BERT_path = "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-6_H-128_A-2/2"

        model_w_BERT = create_BERT_coattention_model(BERT_path, preprocess_path, n_classes=5002, lr=1e-3)

        #BERT model uses sparse categorical crossentropy so no reason to 1-hot-ify the y

        model_w_BERT.fit([df_train_1.context, df_train_1.question],
                df_train_1.answer_start, batch_size=BATCH_SIZE,
                epochs=EPOCHS, validation_data=([df_val_1.context, df_val_1.question], df_val_1.answer_start))
        
        model_w_BERT.save(model_save_path)

    elif (model_choice == 'smooth_bert'):

        preprocess_path = "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3"
        BERT_path = "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-6_H-128_A-2/2"    

        #Running model after smoothing imbalance and binning text
        df_train_1 = undersampling(df_train_1, 'answer_start')

        df_train_1 = text_binning(df_train_1, 'answer_start', 'answer_bin', 10)
        df_val_1 = text_binning(df_val_1, 'answer_start', 'answer_bin', 10)


        smooth_model_w_BERT = create_BERT_coattention_model(BERT_path, preprocess_path, n_classes=501, lr=1e-5)

        smooth_model_w_BERT.fit([df_train_1.context, df_train_1.question],
                df_train_1.answer_bin, batch_size=128,
                epochs=EPOCHS, validation_data=([df_val_1.context, df_val_1.question], df_val_1.answer_bin))
        
        smooth_model_w_BERT.save(model_save_path)



def run(df_train, df_test, model_choice, model_save_path):

    model = tf.keras.models.load_model(model_save_path)

    if model_choice == 'fasttext':

        df_test = df_test[:10]
        MAX_WORDS = 100000
        MAX_SEQ = 2000

        _, test_data, _, test_data_q, _ = tokenizer_preprocessing(df_train, df_test, MAX_WORDS, MAX_SEQ)

        preds = model.predict([test_data, test_data_q])
        preds_hard = np.array([x.argmax() for x in preds])

        acc = sklearn.metrics.accuracy_score(preds_hard, df_test.answer_start.apply(lambda x: int(x)))
        loss = tf.keras.metrics.sparse_categorical_crossentropy(
                                        df_test.answer_start, preds, from_logits=False, axis=-1, ignore_class=None
                                    )

        print("The metrics on the evaluation set are:")
        print(model.metrics_names)
        print(np.mean(loss), acc)

        np.save("./fasttext_coattention_preds.npy",preds)

    elif model_choice == "big_bert":

        preds = model.predict([df_test.context, df_test.question])
        preds_hard = np.array([x.argmax() for x in preds])

        acc = sklearn.metrics.accuracy_score(preds_hard, df_test.answer_start.apply(lambda x: int(x)))
        loss = tf.keras.metrics.sparse_categorical_crossentropy(
                                        df_test.answer_start, preds, from_logits=False, axis=-1, ignore_class=None
                                    )

        print("The metrics on the evaluation set are:")
        print(model.metrics_names)
        print(np.mean(loss), acc)

        np.save("./bert_coattention_preds.npy",preds)

    elif model_choice == "smooth_bert":

        df_test = text_binning(df_test, 'answer_start', 'answer_bin', 10)

        preds = model.predict([df_test.context, df_test.question])
        preds_hard = np.array([x.argmax() for x in preds])

        acc = sklearn.metrics.accuracy_score(preds_hard, df_test.answer_bin.apply(lambda x: int(x)))
        loss = tf.keras.metrics.sparse_categorical_crossentropy(
                                        df_test.answer_bin, preds, from_logits=False, axis=-1, ignore_class=None
                                    )

        print("The metrics on the evaluation set are:")
        print(model.metrics_names)
        print(np.mean(loss), acc)

        np.save("./small_bert_coattention_preds.npy", preds)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--action', type=str, dest='action', help='training or testing options', choices=["train", "run", "train_run"], default="train_run")
    parser.add_argument('--train-path', type=str, dest='train_path', help='Path to training JSON file', default="squad.json")
    parser.add_argument('--dev-path', type=str, dest='dev_path', help='Path to development JSON file', default="squad_dev.json")
    parser.add_argument('--model', type=str, dest='model_choice', help='Model to train', choices=["fasttext", "big_bert", "smooth_bert"], default="fasttext")
    parser.add_argument('--model_save_path', type=str, dest='model_save_path', help='Path to save model', default="./q_a_model")
    options = parser.parse_args()

    action = options.action
    train_path = options.train_path
    dev_path = options.dev_path
    model_choice = options.model_choice
    model_save_path = options.model_save_path
    
    if options.action=="train":
        df_train, _ = prepare(train_path, dev_path)
        train(df_train, model_choice, model_save_path)
    elif options.action=="run":
        df_train, df_test = prepare(train_path, dev_path)
        run(df_train, df_test, model_choice, model_save_path)
    elif options.action=="train_run":
        df_train, df_test = prepare(train_path, dev_path)
        train(df_train, model_choice, model_save_path)
        run(df_train, df_test, model_choice, model_save_path)


    