import pickle
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report

# read pickles and print each stock to csv file

def read_pickle_and_print_to_csv(method):
    pickling = method + ".pickle"

    X = pd.read_pickle(pickling)
    X.sort_values(by=["Y_test_Probs"], ascending=False, inplace=True)
    top25 = X.head(25)

    X = X[X.iloc[:, -2] == 1]
    # write all stocks to csv file

    X.to_csv(method + ".csv")

print("Printing portfolios for each classifier: ")

print()
print("LOG return: ")
read_pickle_and_print_to_csv("X_testLog")
print()

print()
print("RF return: ")
read_pickle_and_print_to_csv("X_test_RF")
print()

print()
print("KNN return: ")
read_pickle_and_print_to_csv("X_test_KNN")
print()

print()
print("EGB return: ")
read_pickle_and_print_to_csv("X_test_EGB")
print()

print()
print("MLP return: ")
read_pickle_and_print_to_csv("X_testMLP")
print()

