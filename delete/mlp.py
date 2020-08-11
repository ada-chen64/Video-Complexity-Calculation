from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import openpyxl
import numpy as np
import os
import matplotlib.pyplot as pt
import pandas as pd

def MLP_predict(X_train, y_train, x_test, y_test):

    clf = MLPClassifier(max_iter= 5000, batch_size=50)
    
    # X_train, x_test, train_label, y_test = train_test_split(data[:,1:], data[:,0],test_size=0.1)

    clf.fit(X_train, y_train)

    print("train")
    # clf.predict(X_train[:,:])
    print(clf.score(X_train, y_train))
    print("test")
    #print(clf.predict_proba(x_test[:1]))
    prediction = clf.predict(x_test[:, :])
    print(prediction)
    print(y_test)
    score = 0
    for i in range(len(prediction)):
        # print(prediction[i], y_test[i])
        if prediction[i] == y_test[i]:
            score += 5
        elif abs(prediction[i] - y_test[i]) == 1:
            score += 5
        # elif abs(prediction[i] - y_test[i]) == 2:
        #     score += 2

    print("score: ",score / (len(prediction) * 5))

    print("accuracy: ",clf.score(x_test, y_test))
    return prediction