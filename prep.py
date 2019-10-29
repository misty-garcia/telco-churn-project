import numpy as np

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

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

def encode_hot(train, test, col_name):
    encoded_values = sorted(list(train[col_name].unique()))

    # Integer Encoding
    int_encoder = LabelEncoder()
    train.encoded = int_encoder.fit_transform(train[col_name])
    test.encoded = int_encoder.transform(test[col_name])

    # create 2D np arrays of the encoded variable (in train and test)
    train_array = np.array(train.encoded).reshape(len(train.encoded),1)
    test_array = np.array(test.encoded).reshape(len(test.encoded),1)

    # One Hot Encoding
    ohe = OneHotEncoder(sparse=False, categories='auto')
    train_ohe = ohe.fit_transform(train_array)
    test_ohe = ohe.transform(test_array)

    # Turn the array of new values into a data frame with columns names being the values
    # and index matching that of train/test
    # then merge the new dataframe with the existing train/test dataframe
    train_encoded = pd.DataFrame(data=train_ohe, columns=encoded_values, index=train.index)
    train = train.join(train_encoded)

    test_encoded = pd.DataFrame(data=test_ohe, columns=encoded_values, index=test.index)
    test = test.join(test_encoded)

    return train, test