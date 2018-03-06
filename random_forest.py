'''
Bryan Quah, Matthew Xu
CSE 415 Project
Random forest implementation using decision trees
'''

import decision_tree as dt
import random as r


class RandomForest:
    def __init__(data, n_trees=20, sampling_percentage=0.7):
        self.N_TREES = n_trees
        self.S_PERCENT = sampling_percent # percentage of data to use for each tree
        self.trees = self.__train_forest(data)


    def __train_forest(data):
        '''
        Returns an array of n_trees decision trees.
        '''
        trees = []
        sub_sample_size = len(data) * self.S_PERCENT

        for n in range(self.N_TREES):
            trees.append(dt.build_decision_tree(r.sample(data, sub_sample_size)))

        return trees


    def classify(row, label=True):
        '''
        Aggregates the results from the decision trees on the given row.
        '''
        res = {}
        for tree in self.trees:
            res_label = dt.classify(tree, row, label)
            if res_label not in res:
                res[res_label] = 0
            res[res_label] += 1
        max_label = None
        max_val = 0
        for k in res.keys():
            if res[k] > max_val:
                max_label = k
        return max_label
