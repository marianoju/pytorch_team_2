"""
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
"""

# import necessary libraries only here
import pandas as pd
from preprocessing import preprocessing
from decision_tree import decision_tree
from evaluation import print_errors

if __name__ == '__main__':

   """ --------------------------------------------------------------------
   Preprocessing takes an input: data
   and returns as output: cleaned_data 
   -------------------------------------------------------------------- """
   data = pd.read_csv("data/housing.csv", na_values='')
   preprocessing(data)

   """ --------------------------------------------------------------------
   Each method takes an input: cleaned_data 
   and returns as output: test, predicted 
   to-do: uncomment method when in place 
   -------------------------------------------------------------------- """
   cleaned_data = pd.read_csv("data/cleaned_data.csv", na_values='')

   decision_tree(cleaned_data)
   # decision_tree_with_pruning(cleaned_data)
   # random_forest(cleaned_data)

   """ --------------------------------------------------------------------
   Each method is evaluated by testing the prediction of the model  
   on a test subset and returns as output: 
   MSE, RMSE, R2, RMSE % of mean, Calibration
   to-do: uncomment method when in place  
   -------------------------------------------------------------------- """
   print_errors(y_test, y_prediction)
   # evaluation.decision_tree_with_pruning(test, predicted)
   # evaluation.random_forest(test, predicted)

   # plot_result(input)
