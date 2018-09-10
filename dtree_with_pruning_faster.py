import copy
import numpy as np
from sklearn.tree import DecisionTreeRegressor
import classifier.prune_faster
from sklearn.metrics import mean_squared_error
import time


def dtree_with_pruning(X_train, X_test, y_train, y_test,*,max_depth=None,
                      random_state=None):

    # Initiate model
    d_tree = DecisionTreeRegressor(max_depth=max_depth,
                                      random_state=random_state)
    dtwp_model = d_tree

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

    return y_test, pred, dtwp_model
