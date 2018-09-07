import pandas as pd
import features

import seaborn as sns
sns.set()
from sklearn.tree import DecisionTreeRegressor


if __name__ == '__main__':
    data = pd.read_csv("data/housing.csv")
    data_cleaned = features.drop_nan(data)
    data_cleaned = features.encode2binary(data_cleaned)
    #print(data)

    #splitting up the dataset





