from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import pandas as pd

def dropnan (data):
        data2=data.dropna(axis=0, how='any')
        return data2

def scaling(method,X_train,X_test,y_test,y_train):
    if method == "Standard":
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
    if method == "MinMax":
        min_max_scaler = MinMaxScaler()
        min_max_scaler.fit(X_train)
        X_train = min_max_scaler.transform(X_train)
        X_test = min_max_scaler.transform(X_test)
    return X_train,y_train,X_test,y_test

def polynomialfeatures(X,degree):
    poly_features = PolynomialFeatures(degree=degree)
    X_poly = poly_features.fit_transform(X)
    return X_poly


def binaerisierung(data, column):
    encoder = OneHotEncoder()
    cat_1hot = encoder.fit_transform(data[column].values.reshape(-1, 1))
    np_cat_1hot = cat_1hot.toarray()
    pd_cat_1hot = pd.DataFrame(np_cat_1hot)
    pd_cat_1hot = pd_cat_1hot.add_prefix(column)
    data = pd.concat([data.reset_index(drop=True), pd_cat_1hot.reset_index(drop=True)],
                     axis=1)
    data = data.drop(column, axis=1)
    return data

def categorical(data):
    # hier wird das kategorische Merkmal verarbeitet
    df = pd.get_dummies(data, drop_first=True)
    return df