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
    dtwp_model = str(dtree) + '\n\nwith Pruning (Legacy) '

    dtwp_fit_start = time.time()
    dtree.fit(X_train, y_train)
    dtwp_fit_end = time.time()
    dtwp_fit_time = dtwp_fit_end - dtwp_fit_start

    dtwp_pred_start = time.time()
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

    dtwp_pred_end = time.time()
    dtwp_pred_time = dtwp_pred_end - dtwp_pred_start

    evaluation.save_errors(y_test, pred, dtwp_model,
                           dtwp_fit_time, dtwp_pred_time)

    return y_test, pred, dtwp_model, dtwp_fit_time, dtwp_pred_time


def dtree_with_pruning_faster(X_train, X_test, y_train, y_test, *,
                              max_depth=None,
                              random_state=None):

    # Initiate model
    dtree = DecisionTreeRegressor(max_depth=max_depth,
                                  random_state=random_state)
    dtwpf_model = str(dtree) + '\n\nwith Pruning (Faster) '

    # Fit model
    dtwpf_fit_start = time.time()
    dtree.fit(X_train, y_train)
    dtwpf_fit_end = time.time()
    dtwpf_fit_time = dtwpf_fit_end - dtwpf_fit_start

    dtwpf_pred_start = time.time()
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

    # uncomment to export errors to CSV file
    # pd.DataFrame(test_errors).to_csv('results/test_errors_dtwpf.csv', index=False) # noqa: E501
    # pd.DataFrame(train_errors).to_csv('results/train_errors_dtwpf.csv', index=False) # noqa: E501

    # Find the best tree based on test data
    test_errors_np = np.array(test_errors)
    index = test_errors_np.argmin()
    pred = tree_pruner.trees[index].predict(X_test)

    dtwpf_pred_end = time.time()
    dtwpf_pred_time = dtwpf_pred_end - dtwpf_pred_start

    evaluation.save_errors(y_test, pred, dtwpf_model,
                           dtwpf_fit_time, dtwpf_pred_time)

    return y_test, pred, dtwpf_model, dtwpf_fit_time, dtwpf_pred_time
