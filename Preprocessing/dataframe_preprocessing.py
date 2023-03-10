import pandas as pd
from sklearn.preprocessing import LabelBinarizer

def Missmatched_NaN_checker(df):
    for i in range(df.shape[0]): #Checking if there is a NaN in one of the columns but a value in the other which would be an issue
        if (df.iloc[i].text != df.iloc[i].text) ^ (df.iloc[i].answer_start != df.iloc[i].answer_start):
            return False

def df_na_handler(df):
    
    df.text = df.text.fillna('UNK')

    df.answer_start = df.answer_start.fillna(5001.0)

def text_preprocessor(dataframe, column_name):
  dataframe[column_name] = dataframe[column_name].apply(lambda x: x.replace('é', 'e'))
  dataframe[column_name] = dataframe[column_name].apply(lambda x: x.replace('-', ' ')) #splitting hyphenated words, as it seems more sensible
  dataframe[column_name] = dataframe[column_name].apply(lambda x: x.lower()) #lowercasing
  return dataframe


def label_binarizer(df_train, y_col):
    lb = LabelBinarizer()
    lb.fit([i for i in range(5002)])
    y_1_hot = lb.transform(df_train[y_col])

    return y_1_hot

