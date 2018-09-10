from unittest import TestCase

import numpy as np

from sklearn.tree import DecisionTreeRegressor

from classifier.prune_faster import prune


class TestPrune(TestCase):

    def set_up_test_tree(self):
        input_data = np.zeros((12, 2))
        self.target = np.zeros((12, 1))


        for i in range(0, 12):
            input_data[i][0] = 0.1 * i
            input_data[i][1] = 1 - 0.1 * i
            self.target[i][0] = np.math.floor(i / 3)

        self.mytree = DecisionTreeRegressor().fit(input_data, self.target)
        prune(self.mytree.tree_, 0)

    def test_n_sample_reset(self):
        self.assertEqual(sum(self.mytree.tree_.n_node_samples), 12)

    def test_children_reset(self):
        self.assertEqual(sum(1 * (self.mytree.tree_.children_left != -1)), 0)
        self.assertEqual(sum(1 * (self.mytree.tree_.children_right != -1)), 0)

    def test_impurity_reset(self):
        mean = 18 / 12
        error = 0
        for i in range(0, 12):
            error += (mean-self.target[i])**2

        self.assertEquals(sum(self.mytree.tree_.impurity), 1/12 * error)

    pass

