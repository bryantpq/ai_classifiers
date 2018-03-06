'''
Bryan Quah, Matthew Xu
CSE 415 Project
'''

import numpy as np

from two_layer_net import TwoLayeredNet
from random_forest import RandomForest

def main():
    print_intro()
    classifier = get_classifier()
    data_set = get_data()
    if classifier == "1":
        use_random_forest(data_set)
    else:
        use_nn(data_set)
        pass

def print_intro():
    print("CSE 415 Project on classifiers by Bryan & Matthew.")
    print("This program selects a classifier and trains it on a selected data set.")
    print("After that, it runs the trained classifier on some testing data and reports the results.")
    print()

def get_classifier():
    print("The available classifiers are a (1) Random Forest and (2) Neural Network.")
    print("Would you like to use the (1) Random Forest or (2) Neural Network?")
    user_clas = input("> ")
    while user_clas != "1" and user_clas != "2":
        print("Please enter \"1\" for Random Forest or \"2\" for Neural Network...")
        user_clas = input("> ")
    print()
    return user_clas

def get_data():
    print("The available data sets are (1) CIFAR-10 and (2) CS:GO Matchmaking data.")
    print("Would you like to use the (1) CIFAR-10 data set or (2) CS:GO Matchmaking data set?")
    user_data = input("> ")
    while user_data != "1" and user_data != "2":
        print("Please enter \"1\" for CIFAR-10 or \"2\" for CS:GO Matching data.")
        user_data = input("> ")
    print()
    return user_data

def use_nn(data):
    input_size = 32 * 32 * 3
    hidden_size = 80
    num_classes = 10
    net = TwoLayeredNet(input_size, hidden_size, num_classes)
    print('Using default parameters to train: 80 hidden nodes, 1500 iterations, 300 batch size, 1e-4 learning rate, 0.95 learning rate decay, and 0.7 regularization strength')
    training_data = aggregate_cifar()
    print(np.array(training_data).shape)
    X = training_data[0:3072][:]
    print(np.array(X).shape)
    net.train()
def use_random_forest(data):
    # TODO:
    # add code to use csgo data instead
    training_data = aggregate_cifar() if data == "1" else aggregate_cifar()
    n_trees = input("How many decision trees would you like to use in your " +\
                    "random forest? Use 1 for a decision tree\n> ")
    while not n_trees.isdigit() or int(n_trees) < 1:
        print("Please enter an integer greater than 1...")
        n_trees = input("> ")
    rf = RandomForest(training_data, n_trees) # create and train random forest
    test_data, test_labels = unpickle("cifar-10-batches-py/test_batch")
    for i in range(10000):
        test_data = np.append(test_data[i], test_labels[i])
    
    pass_count = 0
    fail_count = 0
    for row in test_data:
        if rf.classify(row) == row[-1]:
            pass_count += 1
        else:
            fail_count += 1

    # Report results
    print("Correct classifications: " + str(pass_count))
    print("Wrong classifications: " + str(fail_count))
    print("Accuracy: " + str(float(pass_count) / (pass_count + fail_count)))


def aggregate_cifar(append_label=True):
    '''
    Aggregates the CIFAR-10 data and returns a single array consisting of 50000 arrays
    with RGB values for each image, last value of each array corresponds to a label
    '''
    full_batch = []
    full_label = []
    FILE_NAME = "cifar-10-batches-py/data_batch_"
    FILE_NUM = 5 
    IMAGES_PER_BATCH = 10000

    if append_label:
        for i in range(FILE_NUM):
            batch_data, labels_data = unpickle(FILE_NAME + str(i + 1))
            for j in range(IMAGES_PER_BATCH):
                full_batch.append(np.append(batch_data[j], labels_data[j]))
        return full_batch
    else:
        for i in range(FILE_NUM):
            batch_data, labels_data = unpickle(FILE_NAME + str(i + 1))
            for j in range(IMAGES_PER_BATCH):
                full_batch.append(batch_data[j])
                full_label.append(labels_data[j])
        return full_batch, full_label

def unpickle(file):
    '''
    Unpacks the data files for CIFAR-10
    '''
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='latin1')
    return dict['data'], dict['labels']

if __name__ == "__main__":
    main()
