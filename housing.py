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

# import necessary libraries here 
import pandas as pd
import numpy as np


if __name__ == '__main__':

   # Preprocessing takes a data input 
   # and gives an output: cleaned_data

   preprocessing()

   # Hier werden die verschiedene ML tools durchgeführt und getested
   decision_tree(data_in = data)
   decision_tree_with_pruning(data_in = data)
   random_forest(data_in = data)

   evaluation.decision_tree(input)
   evaluation.decision_tree_with_pruning(input)
   evaluation.random_forest(input)

   plot_result(input)

