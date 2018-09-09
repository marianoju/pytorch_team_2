import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor


def decision_tree(cleaned_data):
    cleaned_data = pd.read_csv("data/cleaned_data.csv", na_values='')

    """ --------------------------------------------------------------------
    we define features and target
    -------------------------------------------------------------------- """
    data_X = cleaned_data.drop("median_house_value", axis=1)  # features
    data_y = cleaned_data["median_house_value"]  # target, or: dependent variable

    """ --------------------------------------------------------------------
    we split the dataset pseudo-randomly into training and testing subsets.
    -------------------------------------------------------------------- """
    X_train, X_test, y_train, y_test = train_test_split(data_X, data_y,
                                                        test_size=0.2,
                                                        random_state=11)

    max_depth = 5
    dTree = DecisionTreeRegressor(max_depth=max_depth)
    dTree.fit(X_train, y_train)
    y_prediction = dTree.predict(X_test)

    return y_test, y_prediction


if __name__ == '__main__':
    print("decision_tree() takes one input: cleaned_data ")
    print("and returns as output: y_test, y_prediction")
