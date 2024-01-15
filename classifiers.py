from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score



def LogReg(X_train, Y_train, X_test):
    clf = LogisticRegression()
    clf.fit(X_train, Y_train)
    X_testLog = X_test

    Y_test_log = clf.predict(X_testLog)
    probs = clf.predict_proba(X_testLog)

    X_testLog["Y_test Log"] = Y_test_log
    X_testLog["Y_test_Probs"] = probs[:, 1]
    print(X_testLog)
    X_testLog.to_pickle('X_testLog.pickle')

## Not used
def MLP(X_train, Y_train, X_test):
    X_test = X_test.drop(columns=['Y_test Log', "Y_test_Probs"], axis=1)

    #Random_state default setting is None, changed to 1
    clf = MLPClassifier(random_state=1)

    clf.fit(X_train.values, Y_train)
    X_testMLP = X_test
    Y_test_MLP = clf.predict(X_testMLP)
    probs = clf.predict_proba(X_testMLP)



    X_testMLP["Y_test MLP"] = Y_test_MLP
    X_testMLP["Y_test_Probs"] = probs[:, 1]
    X_testMLP.to_pickle('X_testMLP.pickle')

def RF(X_train, Y_train, X_test):
    ## Koppla parametrarna
    X_test = X_test.drop(columns=['Y_test Log', "Y_test_Probs"], axis=1)
    """tuned_parameters = {'n_estimators': [32, 256, 512, 1024],
                        'max_features': ['auto', 'sqrt'],
                        'max_depth': [4, 5, 6, 7, 8],
                        'criterion': ['gini', 'entropy']}

    clf = GridSearchCV(RandomForestClassifier(random_state=1),
                       tuned_parameters,
                       n_jobs=6,
                       scoring='precision_weighted',
                       cv=5)"""

    #print('Best score and parameters found on development set:')
    #print()
    #print('%0.3f for %r' % (clf.best_score_, clf.best_params_))
    print("RF")
    # Default settings for max_depth and random_state was None, Max_depth changed to 32 and Random state 1
    clf = RandomForestClassifier(max_depth=None, random_state=1)

    X_test_RF = X_test

    clf.fit(X_train, Y_train)
    probs = clf.predict_proba(X_test_RF)

    Y_test_RF = clf.predict(X_test_RF)




    X_test_RF["Y test Random Forest"] = Y_test_RF
    X_test_RF["Y_test_Probs"] = probs[:, 1]

    print(X_test_RF[X_test_RF.iloc[:, -2] == 1])
    X_test_RF.to_pickle('X_test_RF.pickle')





def extreme_gradient_boosting(X_train, Y_train, X_test):
    X_test = X_test.drop(columns=['Y_test Log', "Y_test_Probs"], axis=1)

    #Default settings except random State which was None, changed to 1
    clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1,
                                    max_depth=3, random_state=1)
    clf.fit(X_train, Y_train)

    X_test_EGB = X_test
    Y_test_EGB = clf.predict(X_test_EGB)
    probs = clf.predict_proba(X_test_EGB)

    #print(classification_report(Y_True, clf.predict(X_test_EGB), target_names=['IGNORE', 'BUY']))

    X_test_EGB["Y test EGB"] = Y_test_EGB
    X_test_EGB["Y_test_Probs"] = probs[:, 1]

    X_test_EGB.to_pickle('X_test_EGB.pickle')


def KNN(X_train, Y_train, X_test):
    X_test = X_test.drop(columns=['Y_test Log', "Y_test_Probs"], axis=1)
    #Default setting: n_neighbour = 5
    neigh = KNeighborsClassifier(n_neighbors=5)
    neigh.fit(X_train, Y_train)
    X_test_KNN = X_test
    Y_test_KNN = neigh.predict(X_test_KNN)
    probs = neigh.predict_proba(X_test_KNN)


    X_test_KNN["Y test KNN"] = Y_test_KNN
    X_test_KNN["Y_test_Probs"] = probs[:, 1]
    X_test_KNN.to_pickle('X_test_KNN.pickle')





