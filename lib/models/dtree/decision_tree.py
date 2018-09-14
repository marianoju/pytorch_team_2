from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
import time
import evaluation


""" --------------------------------------------------------------------
decision_tree() takes input: X_train, X_test, y_train, y_test 
fits DecisionTreeRegressor, computes y_prediction, 
writes results to 'result' and prints 'errors'
-------------------------------------------------------------------- """


def decision_tree(X_train, X_test, y_train, y_test, *, max_depth=None,
                  random_state=None):

    dTree = DecisionTreeRegressor(max_depth=max_depth,
                                  random_state=random_state)

    model = str(dTree) + '\n\nwithout Pruning'

    fit_start = time.time()
    dTree.fit(X_train, y_train)
    fit_end = time.time()
    fit_time = fit_end - fit_start

    pred_start = time.time()
    y_prediction = dTree.predict(X_test)
    pred_end = time.time()
    pred_time = pred_end - pred_start

    evaluation.save_errors(y_test, y_prediction, model, fit_time, pred_time)
    evaluation.print_errors(y_test, y_prediction, model, fit_time, pred_time)


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

    model = str(dTree) + '\n\nwithout Pruning '

    fit_start = time.time()
    dTree.fit(X_train, y_train)
    fit_end = time.time()
    fit_time = fit_end - fit_start

    pred_start = time.time()
    y_prediction = dTree.predict(X_test)
    pred_end = time.time()
    pred_time = pred_end - pred_start

    evaluation.save_errors_classified(
        y_test, y_prediction, model, fit_time, pred_time)
    evaluation.print_errors_classified(
        y_test, y_prediction, model, fit_time, pred_time)


if __name__ == '__main__':
    print('decision_tree() takes input: X_train, X_test, y_train, y_test')
    print('writes results to: result')
    print('and prints: errors')