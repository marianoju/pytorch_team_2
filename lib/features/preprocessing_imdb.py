import pandas as pd
import numpy as np


def load_data():
    file = open('./data/train_data.csv', 'r')
    train_data = [list(map(int, line.split(','))) for line in file]

    file = open('./data/test_data.csv', 'r')
    test_data = [list(map(int, line.split(','))) for line in file]

    train_labels = pd.read_csv('./data/train_labels.csv', header=None)
    test_labels = pd.read_csv('./data/test_labels.csv', header=None)

    x_train = vectorize_sequences(train_data)
    x_test = vectorize_sequences(test_data)

    return x_train, x_test, train_labels, test_labels


def vectorize_sequences(sequences, dimension=10000):
    # Create an all-zero matrix of shape (len(sequences), dimension)
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.  # set specific indices of results[i] to 1s
    return results
