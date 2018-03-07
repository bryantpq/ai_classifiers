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
        trees.append(dt.build_decision_tree(data))

        return trees


    def __train_forest(self, data, sampling_percent):
        '''
        Returns an array of n_trees decision trees.
        '''
        trees = []
        sub_sample_size = int(len(data) * sampling_percent)

        for n in range(self.N_TREES):
            print("Making tree number " + str(n + 1))
            sub_data = r.sample(data, sub_sample_size)
            trees.append(dt.build_decision_tree(sub_data))

        return trees


    def classify(self, row, label=True):
        '''
        Aggregates the results from the decision trees on the given row.
        '''
        agg_res = {}
        for tree in self.trees:
            print("Classifying with a new tree...")
            tree_res = dt.classify(tree, row)
            # TODO
            # tree_res is getting None type
            print(type(tree_res))
            max_label = None
            max_val = 0
            for k in tree_res.keys():
                if tree_res[k] > max_val:
                    max_label = k
                    max_val = res[k]
            print("This tree thinks the object is: " + str(max_label))

            if max_label not in agg_res:
                agg_res[max_label] = 0
            agg_res[max_label] += 1

        max_label = None
        max_val = 0
        for k in agg_res.keys():
            if agg_res[k] > max_val:
                max_label = k
                max_val = agg_res[k]
        return max_label
