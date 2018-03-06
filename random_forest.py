'''
Bryan Quah, Matthew Xu
CSE 415 Project
Random forest implementation using decision trees
'''

import decision_tree as dt
import random as r


class RandomForest:
    def __init__(self, data, n_trees, sampling_percentage=0.7):
        self.N_TREES = int(n_trees)
        self.trees = self.__train_tree(data) if int(n_trees) == 1 else self.__train_forest(data, sampling_percentage)


    def __train_tree(self, data):
        trees = []
        trees.append(dt.build_decision_tree)

        return trees


    def __train_forest(self, data, sampling_percent):
        '''
        Returns an array of n_trees decision trees.
        '''
        trees = []
        sub_sample_size = len(data) * sampling_percent

        for n in range(self.N_TREES):
            trees.append(dt.build_decision_tree(r.sample(data, sub_sample_size)))

        return trees


    def classify(self, row, label=True):
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
