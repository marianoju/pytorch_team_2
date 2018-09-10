import copy
import numpy as np
from sklearn.tree import DecisionTreeRegressor
import classifier.prune
from sklearn.metrics import mean_squared_error
import time


def dtree_with_pruning(X_train, X_test, y_train, y_test,*,max_depth=None,
                      random_state=None):

    # Erstellen und Trainieren des ursprünglichen Baumes

    dtree = DecisionTreeRegressor(max_depth=max_depth,
                                      random_state=random_state)
    dtwp_model = dtree
    dtree.fit(X_train, y_train)

    # Erstellen einer Liste zum Speichern der ge-prunten Bäume
    tree_array = [dtree]
    num_nodes = dtree.tree_.capacity

    # Pruning der Bäume und Anhängen an die Liste
    k = 1

    while num_nodes > 1:
        tree_array.append(copy.deepcopy(tree_array[k - 1]))
        min_node_idx, min_gk = classifier.prune.determine_alpha(tree_array[k].tree_)
        classifier.prune.prune(tree_array[k].tree_, min_node_idx)
        num_nodes = sum(1 * (tree_array[k].tree_.n_node_samples != 0))
        k += 1

    # Finden des besten Baumes, basierend auf den Test-Daten
    predictlist = []

    for i in range(0, len(tree_array)):
        pred = tree_array[i].predict(X_test)
        #predictlist.append(tree_array[i].score(X_test, y_test))
        predictlist.append(mean_squared_error(y_test, pred))

    tree_scores = np.array(predictlist)
    index=tree_scores.argmin()
    pred = tree_array[index].predict(X_test)

    return y_test, pred, dtwp_model

def dtree_with_pruning_faster(X_train, X_test, y_train, y_test,*,max_depth=None,
                      random_state=None):

    # Initiate model
    d_tree = DecisionTreeRegressor(max_depth=max_depth,
                                      random_state=random_state)
    dtwpf_model = d_tree

    # Fit model
    d_tree.fit(X_train, y_train)

    # Pruning trees
    tree_pruner = classifier.prune_faster.TreePruner(d_tree)
    tree_pruner.run()

    # Calculating errors
    test_errors = []
    train_errors = []

    for tree in tree_pruner.trees:
        y_pred_test = tree.predict(X_test)
        test_errors.append(mean_squared_error(y_test, y_pred_test))

        y_pred_train = tree.predict(X_train)
        train_errors.append(mean_squared_error(y_train, y_pred_train))

    #pd.DataFrame(test_errors).to_csv("test_errors.csv", index=False)
    #pd.DataFrame(train_errors).to_csv("train_errors.csv", index=False)

    # Find the best tree based on test data
    test_errors_np = np.array(test_errors)
    index = test_errors_np.argmin()
    pred = tree_pruner.trees[index].predict(X_test)

    return y_test, pred, dtwpf_model
