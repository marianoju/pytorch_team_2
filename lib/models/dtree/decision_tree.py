from sklearn.tree import DecisionTreeRegressor
import time

""" --------------------------------------------------------------------
decision_tree() takes input: X_train, X_test, y_train, 
fits DecisionTreeRegressor and returns as output: y_test, y_prediction 
-------------------------------------------------------------------- """

def decision_tree(X_train, X_test, y_train, y_test, *, max_depth=None,
                  random_state=None):

    dTree = DecisionTreeRegressor(max_depth=max_depth,
                                  random_state=random_state)
    dt_model = str(dTree) + "\n\nwithout Pruning "

    dt_fit_start = time.time()
    dTree.fit(X_train, y_train)
    dt_fit_end = time.time()
    dt_fit_time = dt_fit_end - dt_fit_start

    dt_pred_start = time.time()
    y_prediction = dTree.predict(X_test)
    dt_pred_end = time.time()
    dt_pred_time = dt_pred_end - dt_pred_start

    return y_test, y_prediction, dt_model, dt_fit_time, dt_pred_time


if __name__ == '__main__':
    print("decision_tree() takes input: X_train, X_test, y_train, ")
    print("y_test and returns as output: y_test, y_prediction")
