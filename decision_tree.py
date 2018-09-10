from sklearn.tree import DecisionTreeRegressor


def decision_tree(X_train, X_test, y_train, y_test, *, max_depth=13):

    dTree = DecisionTreeRegressor(max_depth=max_depth)
    dTree.fit(X_train, y_train)
    y_prediction = dTree.predict(X_test)

    return y_test, y_prediction

""" --------------------------------------------------------------------
we concatenate the rows with (scaled) numerical and categorical values 
-------------------------------------------------------------------- """

if __name__ == '__main__':
    print("decision_tree() takes input: X_train, X_test, y_train, ")
    print("y_test and returns as output: y_test, y_prediction")
