#---- required imports
import json
import pandas as pd
import numpy as np
import pickle
import string
import tensorflow as tf
from sklearn.model_selection import train_test_split
import tensorflow_text as tf_text
import tensorflow_hub as tf_hub
import re
import gc
import math
from keras.utils.vis_utils import plot_model
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Bidirectional, Concatenate, Dot, Activation, RepeatVector, Permute, Attention
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from Preprocessing.Json_to_df import *

train_path = "./squad.json"
dev_path = "./squad_dev.json"

df_train = squad_json_to_dataframe_train(train_path)

df_dev =  squad_json_to_dataframe_dev(dev_path)

#Fasttext_embeddings gotten through wget https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.vec.gz

from Preprocessing.Fasttext_embs import *

filename = "cc.en.300.vec"

vocab, vecs = fasttext_reader(filename)

#TO BE CONTINUED