import copy
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from classifier.prune import determine_alpha, prune
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

sns.set()

def dtree_with_pruning(X_train, X_test, y_train, y_test):

   d_tree = DecisionTreeRegressor(random_state=max_depth=13)
   d_tree.fit(X_train, y_train)

   tree_array = [d_tree]

   num_nodes = d_tree.tree_.capacity
   index = 0
   alpha = 0
   k = 1

   while num_nodes > 1:
       tree_array.append(copy.deepcopy(tree_array[k - 1]))

       min_node_idx, min_gk = determine_alpha(tree_array[k].tree_)

       prune(tree_array[k].tree_, min_node_idx)

       num_nodes = sum(1 * (tree_array[k].tree_.n_node_samples != 0))

       k += 1


   if False:
       for k in range(0,len(tree_array)):
           export_graphviz(tree_array[k], out_file='tree' + str(k) + '.dot')


   #print(len(tree_array))

   predictlist = []

   for i in range(0, len(tree_array)):
       pred = tree_array[i].predict(X_test)
       #predictlist.append(tree_array[i].score(X_test, y_test))
       predictlist.append(mean_squared_error(y_test, pred))

   tree_scores = pd.DataFrame()
   tree_scores["MSE"] = np.array(predictlist)

   index=tree_scores.arg_min("MSE")

   pred = tree_array[index].predict(X_test)

   return y_test, pred
