import copy
import numpy as np
from sklearn.tree import DecisionTreeRegressor
import classifier.prune
from sklearn.metrics import mean_squared_error
import time


def dtree_with_pruning(X_train, X_test, y_train, y_test,*,max_depth=None,
                      random_state=None):

    # Erstellen und trainieren des ursprünglichen Baumes

    dtree = DecisionTreeRegressor(max_depth=max_depth,
                                      random_state=random_state)
    dtree.fit(X_train, y_train)

    # Erstellen einer Liste zum Speichern der ge-prunten Bäume
    tree_array = [dtree]
    num_nodes = dtree.tree_.capacity

    # Pruning der Bäume und anhängen an die Liste
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

    return y_test, pred
