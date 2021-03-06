'''
Bryan Quah, Matthew Xu
CSE 415 Project
This is the interface that handles user interaction in a command line interface.
'''

import time
import numpy as np
import pandas as pd

from two_layer_net import TwoLayeredNet
from random_forest import RandomForest

def main():
    print_intro()
    classifier = get_classifier()
    data_set = get_data()
    if data_set == "2":
        print("Using this dataset we take information about how a player is attacking another player in a game to predict the average rank that the match takes place in.\n")

    if classifier == "1":
        use_random_forest(data_set)
    else:
        use_nn(data_set)

def print_intro():
    '''
    Outputs what the program is about.
    '''
    print("CSE 415 Project on classifiers by Bryan & Matthew.")
    print("This program selects a classifier and trains it on a selected data set.")
    print("After that, it runs the trained classifier on some testing data and reports the results.")
    print()

def get_classifier():
    '''
    Gets the type of classifier the user would like to use.
    '''
    print("The available classifiers are a (1) Random Forest and (2) Neural Network.")
    print("Would you like to use the (1) Random Forest or (2) Neural Network?")
    user_clas = input("> ")
    while user_clas != "1" and user_clas != "2":
        print("Please enter \"1\" for Random Forest or \"2\" for Neural Network...")
        user_clas = input("> ")
    print()
    return user_clas

def get_data():
    '''
    Gets which data set the user would like to use.
    '''
    print("The available data sets are (1) CIFAR-10 and (2) CS:GO Matchmaking data.")
    print("Would you like to use the (1) CIFAR-10 data set or (2) CS:GO Matchmaking data set?")
    user_data = input("> ")
    while user_data != "1" and user_data != "2":
        print("Please enter \"1\" for CIFAR-10 or \"2\" for CS:GO Matching data.")
        user_data = input("> ")
    print()
    return user_data

def use_nn(data):
    '''
    Trains and predicts using a neural net on a data set.
    '''
    if data == "1":
        input_size = 32 * 32 * 3
        hidden_size = 80
        num_classes = 10
        net = TwoLayeredNet(input_size, hidden_size, num_classes)
        print(
            'Using tuned parameters to train: 80 hidden nodes, 1500 iterations, 300 batch size, 1e-4 learning rate, 0.95 learning rate decay, and 0.7 regularization strength')
        training_data = aggregate_cifar(False)
        X = np.array(training_data[0])
        y = np.array(training_data[1])

        print("training...")
        start_time = time.time()
        net.train(X, y, learning_rate=1e-4, learning_rate_decay=0.95, reg=5e-6, num_iters=1500, batch_size=300)
        print("Training time: " + str(time.time() - start_time) + " seconds")

        test_data, test_labels = unpickle("cifar-10-batches-py/test_batch")
        pred_y = net.predict(test_data)

        acc = (pred_y == test_labels).mean()
        print("Accuracy: " + str(acc))
    elif data == "2":
        pass
        x_data, y_data = load_csgo()
        x_data = x_data.values
        y_data = y_data.values.astype(int)

        #splits dataset into training and test
        x_train = x_data[:int((len(x_data) + 1) * .80)]
        y_train = y_data[:int((len(y_data) + 1) * .80)]
        x_test = x_data[int(len(x_data) * .80 + 1):]
        y_test = y_data[int(len(y_data) * .80 + 1):]

        input_size = 11
        hidden_size = 50
        #18 ranks in csgo
        num_classes = 18
        net = TwoLayeredNet(input_size, hidden_size, num_classes)

        print("training...")
        start_time = time.time()
        net.train(x_train, y_train, learning_rate=1e-4, learning_rate_decay=0.95, reg=5e-6, num_iters=1500, batch_size=300)
        print("Training time: " + str(time.time() - start_time) + " seconds")

        pred_y = net.predict(x_test)
        acc = (pred_y == y_test).mean()
        print("Accuracy: " + str(acc))


def use_random_forest(data):
    '''
    Trains and predicts using a random forest on a data set.
    '''
    n_trees = input("How many decision trees would you like to use in your " +\
                    "random forest?\nUse 1 for a decision tree\n> ")
    while not n_trees.isdigit() or int(n_trees) < 1:
        print("Please enter an integer greater than 1...")
        n_trees = input("> ")
    print()

    if data == "1": # use cifar
        # Get training data
        n_files = input("How many file batches would you like to use?\nThere are 5.\n> ")
        while not n_files.isdigit() or int(n_files) < 1 or int(n_files) > 5:
            print("Please enter an integer between 1 and 5...")
            n_files = input("> ")
        print()
        n_files = int(n_files)

        n_images = input("How many images would you like from each file?\nThere are 10000 images in each file.\n> ")
        while not n_images.isdigit() or int(n_images) < 1 or int(n_images) > 10000:
            print("Please enter an integer between 1 and 10000...")
            n_images = input("> ")
        print()
        n_images = int(n_images)
        training_data = aggregate_cifar(n_files=n_files, n_images=n_images)

        # Get test data
        test_data, test_labels = unpickle("cifar-10-batches-py/test_batch", n_images=10)
        test_full = np.array([np.append(test_data[0], test_labels[0])])
        for i in range(1, len(test_labels)):
            test_full = np.vstack((test_full, np.append(test_data[i], test_labels[i])))

    else:           # use csgo
        n_data = input("How many rows of data would you like to use for training and testing? 955466 rows available.\n80% will be used for training, 20% for testing\n> ")
        while not n_data.isdigit() or int(n_data) < 1 or int(n_data) > 955466:
            print("Please enter an integer between 1 and 955466...")
            n_data = input("> ")
        print()
        n_data = int(n_data)
        full_data = load_csgo(False)
        full_data = full_data.values
        full_data = full_data[:n_data]

        #splits dataset into training and test
        training_data = full_data[:int((len(full_data) + 1) * .80)]
        test_full = full_data[int(len(full_data) * .80 + 1):]

        print("Done unpacking CS:GO data...")

    start_time = time.time()
    rf = RandomForest(training_data, n_trees) # create and train random forest
    print("Training time: " + str(time.time() - start_time) + " seconds")

    
    pass_count = 0
    fail_count = 0
    print("Classifying test data...")
    for row in test_full:
        res = rf.classify(row, label=True)
        print("Predicted: " + str(res) + "\tActual: " + str(row[-1]))
        if res == row[-1]:
            pass_count += 1
        else:
            fail_count += 1

    # Report results
    print("Correct classifications: " + str(pass_count))
    print("Wrong classifications: " + str(fail_count))
    print("Accuracy: " + str(float(pass_count) * 100/ (pass_count + fail_count)) + "%")


def aggregate_cifar(append_label=True, n_files=5, n_images=None):
    '''
    Aggregates the CIFAR-10 data and returns a single array consisting of 50000 arrays
    with RGB values for each image, last value of each array corresponds to a label
    '''
    full_batch = []
    full_label = []
    FILE_NAME = "cifar-10-batches-py/data_batch_"

    if append_label:
        for i in range(n_files):
            batch_data, labels_data = unpickle(FILE_NAME + str(i + 1), n_images=n_images)
            for j in range(len(labels_data)):
                full_batch.append(np.append(batch_data[j], labels_data[j]))
        return full_batch
    else:
        for i in range(n_files):
            batch_data, labels_data = unpickle(FILE_NAME + str(i + 1), n_images=n_images)
            for j in range(len(labels_data)):
                full_batch.append(batch_data[j])
                full_label.append(labels_data[j])
        return full_batch, full_label


def unpickle(file, n_images=None):
    '''
    Unpacks the data files for CIFAR-10
    '''
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='latin1')
    return dict['data'][:n_images], dict['labels'][:n_images] # if n_images = None, use whole array

def load_csgo(split = True):
    '''
    sets up CSGO data and returns it 
    '''
    csgo_data = pd.read_csv('mm_master_demos.csv')
    wanted_data = csgo_data[['round', 'seconds', 'hp_dmg', 'att_pos_x', 'att_pos_y', 'award', 'vic_pos_x', 'vic_pos_y',
                             'ct_eq_val', 't_eq_val', 'att_rank', 'vic_rank', 'avg_match_rank']]

    if split:
        X = wanted_data.iloc[:, 0:11]
        y = wanted_data.iloc[:, 12]
        return X, y
    else:
        return wanted_data.iloc[:,0:12]

if __name__ == "__main__":
    main()
