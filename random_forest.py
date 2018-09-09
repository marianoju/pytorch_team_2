import copy

import pandas as pd
import seaborn as sns
import features
from sklearn.tree import DecisionTreeRegressor

sns.set()

if __name__ == '__main__':
    data = pd.read_csv("data/housing.csv")
    data_cleaned = features.drop_nan(data)
    print(data)
