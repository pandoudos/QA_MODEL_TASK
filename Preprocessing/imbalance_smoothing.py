from imblearn.under_sampling import RandomUnderSampler
import pandas as pd

def undersampling(df, y_col):
    RUS = RandomUnderSampler(random_state=12, sampling_strategy = {5001 : 250, 0:250})
    x_res, _ = RUS.fit_resample(df, df[y_col])
    return x_res

def text_binning(df, y_col, new_col_name, bin_length):
    df[new_col_name] = df[y_col].apply(lambda x: x // bin_length)
    return df