'''
Bryan Quah, Matthew Xu
CSE 415 Project
'''
from two_layer_net import TwoLayeredNet
from random_forest import RandomForest
def main():
    # 1. determine which classifier
    # 2. determine which data set to train on
    # 3. run on test data
    # 4. report results
    print_intro()
    classifier = get_classifier()
    data_set = get_data()
    
def print_intro():
    print("CSE 415 Project on classifiers by Bryan & Matthew.")
    print("This program selects a classifier and trains it on a selected data set.")
    print("After that, it runs the trained classifier on some testing data and reports the results.")

def get_classifier():
    print("The available classifiers are a (1) Random Forest and (2) Neural Network.")
    print("Would you like to use the (1) Random Forest model or (2) Neural Network model?")
    user_clas = input("> ")
    while user_clas != "1" and user_clas != "2":
        print("Please enter \"1\" for Random Forest or \"2\" for Neural Network...")
        user_clas = input("> ")
    return user_clas

def get_data():
    print("The available data sets are (1) CIFAR-10 and (2) CS:GO Matchmaking data.")
    print("Would you like to use the (1) CIFAR-10 data set or (2) CS:GO Matchmaking data set.")
    user_data = input("> ")
    while user_data != "1" and user_data != "2":
        print("Please enter \"1\" for CIFAR-10 or \"2\" for CS:GO Matching data.")
        user_data = input("> ")
    return user_data

if __name__ == "__main__":
    main()
