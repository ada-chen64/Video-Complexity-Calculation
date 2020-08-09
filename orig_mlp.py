from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import openpyxl
import numpy as np
import os
import matplotlib.pyplot as pt
import pandas as pd

read_file = pd.read_excel (r'new_Norms2.xlsx')
read_file.to_csv (r'new_norm.csv', index = None, header=True)
file = pd.read_csv("new_norm.csv")
data = file.iloc[:,:].values
read_file = pd.read_excel (r'test_norms.xlsx')
read_file.to_csv (r'test_norm.csv', index = None, header=True)
testfile = pd.read_csv("test_norm.csv")
testdata = testfile.iloc[:,:].values

clf = MLPClassifier(max_iter= 5000, batch_size=50)
#randome_state makes result same over and over again
#changing learning_rate didn't seem to affect results much
#batch_size at 50-200 keeps result around 45-50, at 400 it lowers to 41-45


X_train = data[:, 1:]
x_test = testdata[:,1:]
train_label=data[:, 0]
actual_label = testdata[:, 0]
# X_train, x_test, train_label, actual_label = train_test_split(data[:,1:], data[:,0],test_size=0.1)

clf.fit(X_train, train_label)

print("train")
# clf.predict(X_train[:,:])
print(clf.score(X_train, train_label))
print("test")
#print(clf.predict_proba(x_test[:1]))
prediction = clf.predict(x_test[:, :])
print(prediction)
print(actual_label)
score = 0
for i in range(len(prediction)):
    # print(prediction[i], actual_label[i])
    if prediction[i] == actual_label[i]:
        score += 5
    elif abs(prediction[i] - actual_label[i]) == 1:
        score += 5
    # elif abs(prediction[i] - actual_label[i]) == 2:
    #     score += 2

print("score: ",score / (len(prediction) * 5))

print("accuracy: ",clf.score(x_test, actual_label))
# X, y = make_classification(n_samples = 100, random_state=1)
# print(len(X[0]))
# print(len(y))