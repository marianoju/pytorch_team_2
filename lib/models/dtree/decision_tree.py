from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
import time
import evaluation


""" --------------------------------------------------------------------
decision_tree() takes input: X_train, X_test, y_train,
fits DecisionTreeRegressor and returns as output: y_test, y_prediction
-------------------------------------------------------------------- """


def decision_tree(X_train, X_test, y_train, y_test, *, max_depth=None,
                  random_state=None):

    dTree = DecisionTreeRegressor(max_depth=max_depth,
                                  random_state=random_state)
    dt_model = str(dTree) + '\n\nwithout Pruning '

    dt_fit_start = time.time()
    dTree.fit(X_train, y_train)
    dt_fit_end = time.time()
    dt_fit_time = dt_fit_end - dt_fit_start

    dt_pred_start = time.time()
    y_prediction = dTree.predict(X_test)
    dt_pred_end = time.time()
    dt_pred_time = dt_pred_end - dt_pred_start

    evaluation.save_errors(y_test, y_prediction, dt_model,
                           dt_fit_time, dt_pred_time)

    return y_test, y_prediction, dt_model, dt_fit_time, dt_pred_time


def decision_tree_classifier(X_train, X_test, y_train, y_test, *,
                             criterion='gini',
                             splitter='best',
                             max_depth=None,
                             min_samples_split=2,
                             min_samples_leaf=1,
                             min_weight_fraction_leaf=0.0,
                             max_features=None,
                             random_state=None,
                             max_leaf_nodes=None,
                             min_impurity_decrease=0.0,
                             min_impurity_split=None,
                             class_weight=None,
                             presort=False):

    dTree = DecisionTreeClassifier(
        criterion=criterion,
        splitter=splitter,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        min_weight_fraction_leaf=min_weight_fraction_leaf,
        max_features=max_features,
        random_state=random_state,
        max_leaf_nodes=max_leaf_nodes,
        min_impurity_decrease=min_impurity_decrease,
        min_impurity_split=min_impurity_split,
        class_weight=class_weight,
        presort=presort)

    dt_model = str(dTree) + '\n\nwithout Pruning '

    dt_fit_start = time.time()
    dTree.fit(X_train, y_train)
    dt_fit_end = time.time()
    dt_fit_time = dt_fit_end - dt_fit_start

    dt_pred_start = time.time()
    y_prediction = dTree.predict(X_test)
    dt_pred_end = time.time()
    dt_pred_time = dt_pred_end - dt_pred_start

    evaluation.save_errors_classified(
        y_test, y_prediction, dt_model, dt_fit_time, dt_pred_time)

    return y_test, y_prediction, dt_model, dt_fit_time, dt_pred_time


if __name__ == '__main__':
    print('decision_tree() takes input: X_train, X_test, y_train, ')
    print('y_test and returns as output: y_test, y_prediction, ')
    print('dt_model, dt_fit_time, dt_pred_time')
