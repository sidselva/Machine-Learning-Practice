#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 24 16:43:48 2020

@author: siddhartha
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import datetime
from sklearn import model_selection
from sklearn import svm
from sklearn import ensemble

begin_time = datetime.datetime.now()
#IMPORT DATA, SPLIT INDEPENDENT & DEPENDENT VARIABLES, CREATE TEST/TRAIN SETS
csv = pd.read_csv("data.csv", delimiter=",")
abs_csv = csv.loc[csv['material']=='abs']
abs_hex = abs_csv.loc[csv['infill_pattern']=='honeycomb']
abs_hex = abs_hex.drop(columns=['material','infill_pattern'])
print(abs_hex)
# print(csv.describe())

#print(csv['material'])

# row = 1
# ABS csv
# for x in csv['material']:
#     row = row+1
#     print(x,row)
#     if x = 'abs':
        
x = abs_hex.drop(['elongation','tensile strength','roughness'],axis=1)
# print(x)
roughness = abs_hex['roughness']
UTS = abs_hex['tensile strength']
elongation = abs_hex['elongation']

xtrain, xtest, roughnesstrain, roughnesstest = sklearn.model_selection.train_test_split(x,roughness,test_size=0.2)
xtrain, xtest, UTStrain, UTStest = sklearn.model_selection.train_test_split(x,UTS,test_size=0.2)
xtrain, xtest, elongationtrain, elongationtest = sklearn.model_selection.train_test_split(x,elongation,test_size=0.2)

abs_hex.hist(bins=15,density=True,figsize=(10,10))

abs_hex.plot(kind='density', subplots = 'True', layout=(4,3), figsize=(10,10), sharex=False)
plt.show()

corrMatrix = abs_hex.corr()
plt.figure(figsize=(10,10))
sns.heatmap(corrMatrix, annot=True)

models = []
models.append(('SupportVectorClassifier', sklearn.svm.SVC()))
models.append(('StochasticGradientDescentC', sklearn.linear_model.SGDClassifier()))
models.append(('RandomForestClassifier', sklearn.ensemble.RandomForestClassifier()))
models.append(('DecisionTreeClassifier', sklearn.tree.DecisionTreeClassifier()))
#models.append(('GaussianNB', sklearn.naive_bayes.GaussianNB()))
models.append(('KNeighborsClassifier', sklearn.neighbors.KNeighborsClassifier()))
models.append(('AdaBoostClassifier', sklearn.ensemble.AdaBoostClassifier()))
#models.append(('LogisticRegression', sklearn.linear_model.LogisticRegression()))

results = []
names = []
seed = 7
scoring = 'accuracy'
for name, model in models:
    kfold = model_selection.KFold(n_splits=8, random_state=seed, shuffle=(True))
    cv_results = model_selection.cross_val_score(model, xtrain, roughnesstrain, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
    
forest = sklearn.ensemble.RandomForestClassifier(n_estimators = 100)
forest.fit(xtrain,roughnesstrain)
ypred = forest.predict(xtest)
from sklearn import metrics
print ("Accuracy of Random Forest is : ", metrics.accuracy_score(roughnesstest, ypred))
forestcv = sklearn.model_selection.cross_validate(forest,xtrain,roughnesstrain,cv=2)
print('20-fold average accuracy of RF is: ',forestcv['test_score'].mean())