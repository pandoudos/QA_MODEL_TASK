def Missmatched_NaN_checker(df):
    for i in range(df.shape[0]): #Checking if there is a NaN in one of the columns but a value in the other which would be an issue
        if (df.iloc[i].text != df.iloc[i].text) ^ (df.iloc[i].answer_start != df.iloc[i].answer_start):
            return False

def df_na_handler(df):
    
    df.text = df.text.fillna('UNK')

    df.answer_start = df.answer_start.fillna(5001.0)

def text_preprocessor(dataframe,column_name):
  dataframe[column_name] = dataframe[column_name].apply(lambda x: x.replace('é','e')) #keeping Beyonce's e
  dataframe[column_name] = dataframe[column_name].apply(lambda x: x.replace('-',' ')) #splitting hyphenated words, as it seems more sensible
  dataframe[column_name] = dataframe[column_name].apply(lambda x: re.sub(r'[^a-zA-Z0-9 ]+', '', x)) #removing non alphanumerical characters except spaces
  dataframe[column_name] = dataframe[column_name].apply(lambda x: x.lower()) #lowercasing
  return dataframe

def tokenizer_preprocessing(df_train, df_val, max_words, max_seq):
    tokenizer = Tokenizer(num_words=max_words,oov_token='__UNK__')
    tokenizer.fit_on_texts([x for x in df_train.context])
    word_index = tokenizer.word_index
    
    train_seqs = tokenizer.texts_to_sequences([x for x in df_train.context])
    val_seqs = tokenizer.texts_to_sequences([x for x in df_val.context])
    train_data = pad_sequences(train_seqs, maxlen = max_seq, padding = 'post')
    val_data = pad_sequences(val_seqs, maxlen = max_seq, padding = 'post')

    train_seqs_q = tokenizer.texts_to_sequences([x for x in df_train_1.question])
    val_seqs_q = tokenizer.texts_to_sequences([x for x in df_val_1.question])
    train_data_q = pad_sequences(train_seqs_q, maxlen = max_seq, padding = 'post')
    val_data_q = pad_sequences(val_seqs_q, maxlen = max_seq, padding = 'post')

    return train_data, val_data, train_data_q, val_data_q