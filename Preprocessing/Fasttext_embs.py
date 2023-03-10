import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer

def fasttext_reader(filename):
    index = 0
    vocab = {}
    with open(filename, 'r', encoding="utf-8", newline='\n',errors='ignore') as f: #Opening the downloaded embedding file
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

def embedding_matrix_maker(word_index, vocab, vecs, max_words):
    embedding_matrix = np.zeros((max_words+2, vecs.shape[0]))
    for word, i in word_index.items():
        try:
            embedding_vector = vecs[vocab[word],:]
            embedding_matrix[i] = embedding_vector
        except:
            pass  
    return embedding_matrix       