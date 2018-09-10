from sklearn.tree import DecisionTreeRegressor

""" --------------------------------------------------------------------
decision_tree() takes input: X_train, X_test, y_train, 
fits DecisionTreeRegressor and returns as output: y_test, y_prediction 
-------------------------------------------------------------------- """

def decision_tree(X_train, X_test, y_train, y_test, *, max_depth=None,
                  random_state=None):

    dTree = DecisionTreeRegressor(max_depth=max_depth,
                                  random_state=random_state)
    dt_model = dTree
    dTree.fit(X_train, y_train)
    y_prediction = dTree.predict(X_test)

    return y_test, y_prediction, dt_model


if __name__ == '__main__':
    print("decision_tree() takes input: X_train, X_test, y_train, ")
    print("y_test and returns as output: y_test, y_prediction")
