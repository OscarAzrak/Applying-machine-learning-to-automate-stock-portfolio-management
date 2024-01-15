import pickle
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report

pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.max_rows', 1500)

"""TRAINING PART"""
x_matrix = pd.read_pickle('x_matrix.pickle')
x_matrix = x_matrix.fillna(0)
x_matrix = x_matrix.replace([np.inf, -np.inf], 0).dropna(how="all")

Y_train = []
for i in x_matrix["YoY"]:
    if i > 0.25:
        Y_train.append(1)
    else:
        Y_train.append(0)

X_train = x_matrix.drop(columns=['YoY', "Sigmoid"], axis=1)




"""TESTING"""

x_test = pd.read_pickle('X_test.pickle')
x_test = x_test.fillna(0)
x_test = x_test.replace([np.inf, -np.inf], 0).dropna(how="all")
Y_test = x_test["YoY"]
Y_True = []
for ret in x_test["YoY"]:
    if round(ret, 2) >=0.25:
        Y_True.append(1)
    else:
        Y_True.append(0)



x_test = x_test.drop(columns=['YoY', "Sigmoid"], axis=1)

clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,
                                 max_depth=1, random_state=0)
clf.fit(X_train, Y_train)

Te = pd.read_pickle('X_test_EGB.pickle').iloc[:,-1].tolist()
#print(Te)
#print(len(Te))
k = 0
for i in Te:
    if i == 0:
        k+=1
#print(k)
print(Y_True)
print(classification_report(Y_True, clf.predict(x_test), target_names=['IGNORE', 'BUY']))
