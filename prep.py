import numpy as np

from sklearn.preprocessing import LabelEncoder

def clean_data(df):
    # for total charges
    # strip white spaces away, replace with NaN, and change variable to float
    df.total_charges = df.total_charges.str.strip().replace("", np.nan).astype(float)
    # drop nulls
    df.dropna(inplace=True)
    return df

def encode(train, test, col_name):
    int_encoder = LabelEncoder()
    train[col_name] = int_encoder.fit_transform(train[col_name])
    test[col_name] = int_encoder.transform(test[col_name])
    return train, test

def encode_add_column(train, test, col_name):
    int_encoder = LabelEncoder()
    train[col_name + "_encoded"] = int_encoder.fit_transform(train[col_name])
    test[col_name + "_encoded"] = int_encoder.transform(test[col_name])
    return train, test