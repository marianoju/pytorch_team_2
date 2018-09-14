import copy
import numpy as np
from sklearn.tree import DecisionTreeRegressor
import models.dtree.prune
import models.dtree.prune_faster
from sklearn.metrics import mean_squared_error
import time
import evaluation


def dtree_with_pruning(X_train, X_test, y_train, y_test, *, max_depth=None,
                       random_state=None):

    # Erstellen und Trainieren des ursprünglichen Baumes

    dtree = DecisionTreeRegressor(max_depth=max_depth,
                                  random_state=random_state)

    model = str(dtree) + '\n\nwith Pruning (Legacy)'

    fit_start = time.time()
    dtree.fit(X_train, y_train)
    fit_end = time.time()
    fit_time = fit_end - fit_start

    pred_start = time.time()
    # Erstellen einer Liste zum Speichern der ge-prunten Bäume
    tree_array = [dtree]
    num_nodes = dtree.tree_.capacity

    # Pruning der Bäume und Anhängen an die Liste
    k = 1

    while num_nodes > 1:
        tree_array.append(copy.deepcopy(tree_array[k - 1]))
        min_node_idx, min_gk = models.dtree.prune.determine_alpha(
            tree_array[k].tree_)
        models.dtree.prune.prune(tree_array[k].tree_, min_node_idx)
        num_nodes = sum(1 * (tree_array[k].tree_.n_node_samples != 0))
        k += 1

    # Finden des besten Baumes, basierend auf den Test-Daten
    predictlist = []

    for i in range(0, len(tree_array)):
        pred = tree_array[i].predict(X_test)
        # predictlist.append(tree_array[i].score(X_test, y_test))
        predictlist.append(mean_squared_error(y_test, pred))

    tree_scores = np.array(predictlist)
    index = tree_scores.argmin()
    pred = tree_array[index].predict(X_test)

    pred_end = time.time()
    pred_time = pred_end - pred_start

    evaluation.save_errors(y_test, pred, model,
                           fit_time, pred_time)
    evaluation.print_errors(y_test, pred, model,
                            fit_time, pred_time)


def dtree_with_pruning_faster(X_train, X_test, y_train, y_test, *,
                              max_depth=None,
                              random_state=None):

    # Initiate model
    dtree = DecisionTreeRegressor(max_depth=max_depth,
                                  random_state=random_state)
    model = str(dtree) + '\n\nwith Pruning (Faster) '

    # Fit model
    fit_start = time.time()
    dtree.fit(X_train, y_train)
    fit_end = time.time()
    fit_time = fit_end - fit_start

    pred_start = time.time()
    # Pruning trees
    tree_pruner = models.dtree.prune_faster.TreePruner(dtree)
    tree_pruner.run()

    # Calculating errors
    test_errors = []
    train_errors = []

    for tree in tree_pruner.trees:
        y_pred_test = tree.predict(X_test)
        test_errors.append(mean_squared_error(y_test, y_pred_test))
        y_pred_train = tree.predict(X_train)
        train_errors.append(mean_squared_error(y_train, y_pred_train))

    # Find the best tree based on test data
    test_errors_np = np.array(test_errors)
    index = test_errors_np.argmin()
    pred = tree_pruner.trees[index].predict(X_test)

    pred_end = time.time()
    pred_time = pred_end - pred_start

    evaluation.save_errors(y_test, pred, model,
                           fit_time, pred_time)
    evaluation.print_errors(y_test, pred, model,
                            fit_time, pred_time)
