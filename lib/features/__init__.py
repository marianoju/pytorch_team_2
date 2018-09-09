from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import OneHotEncoder

import numpy as np
import pandas as pd


def drop_nan(data):
        data=data.dropna(axis=0, how='any')
        return data


def scaling(method,X_train,X_test):
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
    return X_train,X_test


def polynomial_features(X,degree):
    poly_features = PolynomialFeatures(degree=degree)
    X_poly = poly_features.fit_transform(X)
    return X_poly


def reencode_categorical(data):
    # hier wird das kategorische Merkmal verarbeitet
    data = pd.get_dummies(data, drop_first=True)
    return data


def reencode_to_binary(data, column):
    encoder = OneHotEncoder()
    cat_1hot = encoder.fit_transform(data[column].values.reshape(-1, 1))
    np_cat_1hot = cat_1hot.toarray()
    pd_cat_1hot = pd.DataFrame(np_cat_1hot)
    pd_cat_1hot = pd_cat_1hot.add_prefix(column)
    data = pd.concat([data.reset_index(drop=True), pd_cat_1hot.reset_index(drop=True)],
                     axis=1)
    data = data.drop(column, axis=1)
    return data

