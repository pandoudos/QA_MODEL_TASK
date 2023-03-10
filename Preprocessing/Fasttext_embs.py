import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def fasttext_reader(filename):
    index = 0
    vocab = {}
    with open(filename, 'r', encoding="utf-8", newline='\n', errors='ignore') as f: #Opening the downloaded embedding file
        for l in f:
            line = l.rstrip().split(' ')
            if index == 0: #The first row of the file contains a list of the form [vocab_size, embed_size]
                vocab_size = int(line[0]) + 2 #+2 for the UNK and PADDING tokens
                dim = int(line[1])
                vecs = np.zeros([vocab_size,dim]) #The size of the embedding matrix
                vocab["__PADDING__"] = 0
                vocab["__UNK__"] = 1
                index = 2
            else:
                vocab[line[0]] = index #The rest of the rows contain a list starting with the embedded word or token followed by the embedding values
                emb = np.array(line[1:]).astype(float) 
                if (emb.shape[0] == dim): #only move on if a proper embedding has been created
                    vecs[index,:] = emb
                    index+=1 
                else:
                    continue

    return vocab, vecs    

def tokenizer_preprocessing(df_train, df_val, max_words, max_seq):
    tokenizer = Tokenizer(num_words=max_words,oov_token='__UNK__')
    tokenizer.fit_on_texts([x for x in df_train.context])
    word_index = tokenizer.word_index
    
    train_seqs = tokenizer.texts_to_sequences([x for x in df_train.context])
    val_seqs = tokenizer.texts_to_sequences([x for x in df_val.context])
    train_data = pad_sequences(train_seqs, maxlen = max_seq, padding = 'post')
    val_data = pad_sequences(val_seqs, maxlen = max_seq, padding = 'post')

    train_seqs_q = tokenizer.texts_to_sequences([x for x in df_train.question])
    val_seqs_q = tokenizer.texts_to_sequences([x for x in df_val.question])
    train_data_q = pad_sequences(train_seqs_q, maxlen = max_seq, padding = 'post')
    val_data_q = pad_sequences(val_seqs_q, maxlen = max_seq, padding = 'post')

    return train_data, val_data, train_data_q, val_data_q, word_index  

def embedding_matrix_maker(word_index, vocab, vecs, max_words):
    embedding_matrix = np.zeros((max_words+2, vecs.shape[0]))
    for word, i in word_index.items():
        try:
            embedding_vector = vecs[vocab[word],:]
            embedding_matrix[i] = embedding_vector
        except:
            pass  
    return embedding_matrix       