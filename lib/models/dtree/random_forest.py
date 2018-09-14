from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import time
import evaluation

""" --------------------------------------------------------------------
random_forest() takes input: X_train, X_test, y_train, y_test
fits RandomForestRegressor and returns as output: y_test, y_prediction
-------------------------------------------------------------------- """


def random_forest(X_train, X_test, y_train, y_test, *, n_estimators=10,
                  criterion='mse', max_depth=None, min_samples_split=2,
                  min_samples_leaf=1, min_weight_fraction_leaf=0.0,
                  max_features='auto', max_leaf_nodes=None,
                  min_impurity_decrease=0.0, min_impurity_split=None,
                  bootstrap=True, oob_score=False, n_jobs=1,
                  random_state=None, verbose=0, warm_start=False):

    regr = RandomForestRegressor(
        n_estimators=n_estimators,
        criterion=criterion, max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        min_weight_fraction_leaf=min_weight_fraction_leaf,
        max_features=max_features,
        max_leaf_nodes=max_leaf_nodes,
        min_impurity_decrease=min_impurity_decrease,
        min_impurity_split=min_impurity_split,
        bootstrap=bootstrap, oob_score=oob_score,
        n_jobs=n_jobs, random_state=random_state,
        verbose=verbose, warm_start=warm_start)

    model = str(regr)

    fit_start = time.time()
    regr.fit(X_train, y_train)
    fit_end = time.time()
    fit_time = fit_end - fit_start

    pred_start = time.time()
    y_prediction = regr.predict(X_test)
    pred_end = time.time()
    pred_time = pred_end - pred_start

    evaluation.save_errors(y_test, y_prediction,
                           model, fit_time, pred_time)

    evaluation.print_errors(y_test, y_prediction,
                            model, fit_time, pred_time)


def random_forest_classifier(X_train, X_test, y_train, y_test, *,
                             n_estimators=10,
                             criterion='gini',
                             max_depth=None,
                             min_samples_split=2,
                             min_samples_leaf=1,
                             min_weight_fraction_leaf=0.0,
                             max_features='auto',
                             max_leaf_nodes=None,
                             min_impurity_decrease=0.0,
                             min_impurity_split=None,
                             bootstrap=True,
                             oob_score=False,
                             n_jobs=1,
                             random_state=None,
                             verbose=0,
                             warm_start=False,
                             class_weight=None):

    regr = RandomForestClassifier(
        n_estimators=n_estimators,
        criterion=criterion,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        min_weight_fraction_leaf=min_weight_fraction_leaf,
        max_features=max_features,
        max_leaf_nodes=max_leaf_nodes,
        min_impurity_decrease=min_impurity_decrease,
        min_impurity_split=min_impurity_split,
        bootstrap=bootstrap,
        oob_score=oob_score,
        n_jobs=n_jobs,
        random_state=random_state,
        verbose=verbose,
        warm_start=warm_start,
        class_weight=class_weight)

    model = str(regr)

    fit_start = time.time()
    regr.fit(X_train, y_train)
    fit_end = time.time()
    fit_time = fit_end - fit_start

    pred_start = time.time()
    y_prediction = regr.predict(X_test)
    pred_end = time.time()
    pred_time = pred_end - pred_start

    evaluation.save_errors_classified(
        y_test, y_prediction, model, fit_time, pred_time)
    evaluation.print_errors_classified(
        y_test, y_prediction, model, fit_time, pred_time)


if __name__ == '__main__':
    print('random_forest() takes input: X_train, X_test, y_train, ')
    print('fits RandomForestRegressor and returns as output: ' +
          'y_test, y_prediction')
