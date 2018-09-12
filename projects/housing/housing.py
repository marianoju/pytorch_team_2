""" ---------------------------------------------------------------------
The dataset [1] used for this excercise in regression analysis is a
modified version of the California Housing dataset available from Luís 
Torgo [2]. Luís Torgo obtained it from the StatLib repository (which is 
closed now). The dataset may also be downloaded from StatLib mirrors.

This dataset appeared in a 1997 paper titled Sparse Spatial 
Autoregressions by Pace, R. Kelley and Ronald Barry, published in the 
Statistics and Probability Letters journal. They built it using the 
1990 California census data. It contains one row per census block group.
A block group is the smallest geographical unit for which the U.S. 
Census Bureau publishes sample data (a block group typically has a 
population of 600 to 3,000 people). 

[1] https://github.com/ageron/handson-ml/tree/master/datasets/housing
[2] http://www.dcc.fc.up.pt/%7Eltorgo/Regression/cal_housing.html
--------------------------------------------------------------------- """

# import necessary libraries only here
from features.preprocessing_housing import preprocessing_housing
from models.decision_tree import decision_tree
from models.random_forest import random_forest
from models.dtree_with_pruning import dtree_with_pruning, dtree_with_pruning_faster
from evaluation import print_errors
from models.svm import svm_regression


if __name__ == '__main__':
    """ -----------------------------------------------------------------
    Preprocessing takes an input: data
    and returns as output: X_train, X_test, y_train, y_test 
    ----------------------------------------------------------------- """
    X_train, X_test, y_train, y_test = preprocessing_housing()

    """ -----------------------------------------------------------------
    Each method takes an input: X_train, X_test, y_train, y_test
    and returns as output: test, predicted 
    to-do: uncomment method when in place 
    ----------------------------------------------------------------- """

    y_test, y_prediction, dt_model, dt_fit_time, dt_pred_time = decision_tree(
        X_train, X_test, y_train, y_test, max_depth=3, random_state=11)
    y_test, dtwp_y_prediction, dtwp_model, dtwp_fit_time, dtwp_pred_time = dtree_with_pruning(
        X_train, X_test, y_train, y_test, max_depth=3,random_state=11)
    y_test, dtwpf_y_prediction, dtwpf_model, dtwpf_fit_time, dtwpf_pred_time = dtree_with_pruning_faster(
        X_train, X_test, y_train, y_test, max_depth=3,random_state=11)
    y_test, rf_y_prediction, rf_model, rf_fit_time, rf_pred_time = random_forest(
        X_train, X_test, y_train, y_test, max_depth=3, random_state=11, n_estimators=40,
        min_samples_leaf=3, warm_start=True)
    y_test, svmr_y_prediction, svmr_model, svmr_fit_time, svmr_pred_time = svm_regression(
        X_train, X_test, y_train, y_test)

    """ -----------------------------------------------------------------
    Each method is evaluated by testing the prediction of the model  
    on a test subset and returns as output: 
    MSE, RMSE, R2, RMSE % of mean, Calibration
    to-do: uncomment method when in place  
    ----------------------------------------------------------------- """
    dt_errors = print_errors(y_test, y_prediction, dt_model, dt_fit_time, dt_pred_time)
    dtwp_errors = print_errors(y_test, dtwp_y_prediction, dtwp_model, dtwp_fit_time, dtwp_pred_time)
    dtwpf_errors = print_errors(y_test, dtwpf_y_prediction, dtwpf_model, dtwpf_fit_time, dtwpf_pred_time)
    rf_errors = print_errors(y_test, rf_y_prediction, rf_model, rf_fit_time, rf_pred_time)
    svmr_errors = print_errors(y_test, svmr_y_prediction, svmr_model, svmr_fit_time, svmr_pred_time)

    """ -----------------------------------------------------------------
    TODO: Errors of each model could be plotted here for visualization. 
    ----------------------------------------------------------------- """

