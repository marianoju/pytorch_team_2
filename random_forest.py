import pandas as pd
import features


if __name__ == '__main__':
    data = pd.read_csv("data/housing.csv")
    data_cleaned = features.drop_nan(data)
    data_cleaned = pd.get_dummies(data_cleaned, drop_first=True)
