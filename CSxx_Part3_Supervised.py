#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 00:11:20 2020

@author: mihalis
"""


print("Unsupervised Learning")
import sklearn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from sklearn import datasets
from sklearn import metrics
from sklearn import cluster
import pandas as pd
from sklearn.preprocessing import scale
import seaborn as sns

print("read data")
br_data0 = pd.read_csv("br_data.csv")

print("show first and last 10 rows")
print(br_data0.head(10))  # check first 10 rows
print(br_data0.tail(10))
print("show shape of data")
print(br_data0.shape)
print("columns")
print(br_data0.columns)
print("checking for nulls")
print(br_data0.isnull())  # check if there are missing values.Returns Boolean [False=no null, True=null]
print(br_data0.isnull().any())  # returns which columns are intact and which not

# last column is NaN so I drop it
print("Dropping column Id and slicing data prior to Supervised Learning")
     
br_data0 = br_data0.drop("id", axis=1)
br_data0.head()
print(br_data0.shape)

Y= br_data0.iloc[:, 0]
print(Y)

subset1= br_data0.iloc[:, [ 21, 22,23,24,25,26,27,28,29,30]]
print(subset1.columns)
print(subset1.shape)
print("Y is" , Y.head)
print("Dropped column-Sliced Data Complete")

print("Convert diagnosis column into numerical data for downstream Supervised learning analysis")
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
Y1 = labelencoder.fit_transform(Y)
print("Diagnosis column after encoding",Y1)

print("Scale data and find unique values of Y that we will use as k ")
from sklearn.preprocessing import scale
scaled_data1 = scale(subset1) # Scaling teh data
print("Scaled Data1", scaled_data1)



from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from pandas import DataFrame
from io import StringIO



print("performing Logistic Rregression")
lm = LogisticRegression(max_iter =500)
X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(scaled_data1, Y1, test_size=0.30)
lm.fit(X_train, Y_train)
lm.predict_proba(X_test)
predicted = lm.predict(X_test)
Metrics_classification = metrics.classification_report(Y_test, predicted)
test2 = pd.DataFrame(StringIO(Metrics_classification))
print(Metrics_classification)
print(metrics.confusion_matrix(Y_test, predicted))


print("Performing Decision Tree Classifier ")

from sklearn.tree import DecisionTreeClassifier
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(scaled_data1, Y1, test_size = 0.30)
model = DecisionTreeClassifier()
model.fit(X_train, y_train)
print(model)
y_pred = model.predict(X_test)
print(metrics.classification_report(y_test,y_pred))
print(metrics.confusion_matrix(y_test,y_pred))



print("Performing KNeighbors Classification")
from sklearn.neighbors import KNeighborsClassifier
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(scaled_data1, Y1, test_size = 0.30)
model = KNeighborsClassifier(n_neighbors = 10)
model.fit(X_train, y_train)
print(model)
y_pred = model.predict(X_test)
print(metrics.classification_report(y_test,y_pred))
print(metrics.confusion_matrix(y_test,y_pred))


print("Performing GaussianNB")
from sklearn.naive_bayes import GaussianNB
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(scaled_data1, Y1, test_size = 0.30)
model = GaussianNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(metrics.classification_report(y_test,y_pred))
print(metrics.confusion_matrix(y_test,y_pred))






print("Testing  classifiers for all data set to see whether the precision value is increased when we have all data set")

br_data2=br_data0.drop("diagnosis", axis =1)
scaled_data2 = scale(br_data2)
n_samples, n_features = br_data0.shape
n_digits = len(np.unique(Y1)) #It asks how many unique numbers== values you have in the dataset and uses that as n clusters"
print(n_digits)



print("Performing KNeighbors Classification")
from sklearn.neighbors import KNeighborsClassifier
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(scaled_data2, Y1, test_size = 0.30)
model = KNeighborsClassifier(n_neighbors = 30)
model.fit(X_train, y_train)
print(model)
y_pred = model.predict(X_test)
print(metrics.classification_report(y_test,y_pred))
print(metrics.confusion_matrix(y_test,y_pred))

print("Performing GaussianNB")
from sklearn.naive_bayes import GaussianNB
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(scaled_data2, Y1, test_size = 0.30)
model = GaussianNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(metrics.classification_report(y_test,y_pred))
print(metrics.confusion_matrix(y_test,y_pred))

print("Performing Decision Tree Classifier ")

from sklearn.tree import DecisionTreeClassifier
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(scaled_data2, Y1, test_size = 0.30)
model = DecisionTreeClassifier()
model.fit(X_train, y_train)
print(model)
y_pred = model.predict(X_test)
print(metrics.classification_report(y_test,y_pred))
print(metrics.confusion_matrix(y_test,y_pred))
