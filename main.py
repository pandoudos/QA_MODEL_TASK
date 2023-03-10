import sys
from argparse import ArgumentParser

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

def main(train_path, dev_path):
    """Downloading and basic preprocessing of the dataframe"""

    df_train = squad_json_to_dataframe_train(train_path)
    df_dev =  squad_json_to_dataframe_dev(dev_path)

    df_train = df_train.drop_duplicates(subset=['question', 'answer_start', 'c_id']).reset_index(drop=True) #removing duplicates

    Missmatched_NaN_checker(df_train)
    df_na_handler(df_train)
    Missmatched_NaN_checker(df_dev)
    df_na_handler(df_dev)

    y_train = label_binarizer(df_train, 'answer_start')

    df_train_1, df_val_1, y_train_1_hot, y_val_1_hot = train_test_split(df_train, y_train, test_size=0.33, random_state=12)

    for col in ['question', 'context', 'text']:
        df_train = text_preprocessor(df_train_1, col)
        df_val = text_preprocessor(df_val_1, col)

    #Downloading Fasttext embeddings and setting them up

    MAX_WORDS = 100000
    MAX_SEQ = 2000
    BATCH_SIZE = 128
    EPOCHS = 5

    if (options.model_choice == 'fasttext'):    
        #Fasttext embeddings gotten through wget https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.vec.gz
        embeddings_filename = "cc.en.300.vec"
        vocab, vecs = fasttext_reader(embeddings_filename)

        train_data, val_data, train_data_q, val_data_q, word_index = tokenizer_preprocessing(df_train, df_val, MAX_WORDS, MAX_SEQ)
        embedding_matrix = embedding_matrix_maker(word_index, vocab, vecs, MAX_WORDS)

        #Running Model with Fasttext
        model = create_fasttext_coattention_model(embedding_matrix)

        history = model.fit([train_data, train_data_q],
                y_train_1_hot, batch_size=BATCH_SIZE,
                epochs=EPOCHS, validation_data=([val_data, val_data_q], y_val_1_hot))
        
        model.save(options.model_save_path)

    elif (options.model_choice == 'big_bert'):   

        #Running Model with BERT sentence embeddings
        preprocess_path = "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3"
        BERT_path = "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-6_H-128_A-2/2"

        model_w_BERT = create_BERT_coattention_model(BERT_path, preprocess_path, n_classes=5002, lr=1e-3)

        history_w_BERT = model_w_BERT.fit([df_train_1.context, df_train_1.question],
                df_train_1.answer_start, batch_size=BATCH_SIZE,
                epochs=EPOCHS, validation_data=([df_val_1.context, df_val_1.question], df_val_1.answer_start))
        
        results = model_w_BERT.evaluate(([df_dev.context, df_dev.question], df_dev.answer_start))
        
        model_w_BERT.save(options.model_save_path)

    elif (options.model_choice == 'smooth_bert'):    


        #Running model after smoothing imbalance and binning text
        df_train_1 = undersampling(df_train_1, 'answer_start')

        df_train_1 = text_binning(df_train_1, 'answer_start', 'answer_bin', 10)
        df_val_1 = text_binning(df_val_1, 'answer_start', 'answer_bin', 10)


        #BERT model uses sparse categorical crossentropy so no reason to 1-hot-ify the y
        smooth_model_w_BERT = create_BERT_coattention_model(BERT_path, preprocess_path, n_classes=501, lr=1e-5)

        history_smooth = smooth_model_w_BERT.fit([df_train_1.context, df_train_1.question],
                df_train_1.answer_bin, batch_size=128,
                epochs=EPOCHS, validation_data=([df_val_1.context, df_val_1.question], df_val_1.bin))
        
        smooth_model_w_BERT.save(options.model_save_path)

        #Evaluating model
        df_dev = text_binning(df_dev, 'answer_start', 'answer_bin', 10)

        results = smooth_model_w_BERT.evaluate(([df_dev.context, df_dev.question], df_dev.answer_bin))

        print(smooth_model_w_BERT.metrics_names)
        print(results)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--train-path', type=str, dest='train_path', help='Path to training JSON file', default="squad.json")
    parser.add_argument('--dev-path', type=str, dest='dev_path', help='Path to development JSON file', default="squad_dev.json")
    parser.add_argument('--model', type=str, dest='model_choice', help='Model to train', choices=["fasttext", "big_bert", "smooth_bert"], default="smooth_bert")
    parser.add_argument('--model-save-path', type=str, dest='model_save_path', help='Path to save model', default="./q_a_model")
    options = parser.parse_args()

    train_path = options.train_path
    dev_path = options.dev_path

    sys.exit(main(train_path, dev_path))