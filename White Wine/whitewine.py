#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 20 15:40:26 2020

@author: siddhartha
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn import model_selection, preprocessing, feature_selection, ensemble, linear_model, metrics, decomposition
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from pprint import pprint
import datetime

begin_time = datetime.datetime.now()
#IMPORT DATA, SPLIT INDEPENDENT & DEPENDENT VARIABLES, CREATE TEST/TRAIN SETS
csv = pd.read_csv("winequality-white.csv", delimiter=";")
#print(csv.head(5))

x = csv.drop("quality", axis=1)
# print("drop")

y = csv['quality']
# print (y)

xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size=0.2)
# print(len(xtest))

#ANALYZE DATA
print(csv.describe())

#HISTOGRAM
# csv.hist(bins = 7,density=True, figsize=(10,10))
# plt.show()

#DENSITY PLOT
# csv.plot(kind='density', subplots = 'True', layout=(4,3), figsize=(10,10), sharex=False)
# plt.show()

#CORRELATION MATRICES
# corrMatrix = csv.corr()
# plt.figure(figsize=(10,10))
# sns.heatmap(corrMatrix, annot=True)

# sns.pairplot(csv, size=(2))
# plt.show()

#BIN QUALITY AS BAD/GOOD
bins = (2,6,10)
group_names = ['bad', 'good']
csv['quality']=pd.cut(csv['quality'], bins=bins, labels=group_names)
label_quality = LabelEncoder()
csv['quality'] = label_quality.fit_transform(csv['quality'])

pprint(csv['quality'])
print(csv['quality'].value_counts()) 

sns.countplot(csv['quality'])
plt.show()

#EVALUATE ML ALGORITHMS
# models = []
# models.append(('SupportVectorClassifier', sklearn.svm.SVC()))
# models.append(('StochasticGradientDescentC', sklearn.linear_model.SGDClassifier()))
# models.append(('RandomForestClassifier', sklearn.ensemble.RandomForestClassifier()))
# models.append(('DecisionTreeClassifier', sklearn.tree.DecisionTreeClassifier()))
# #models.append(('GaussianNB', sklearn.naive_bayes.GaussianNB()))
# models.append(('KNeighborsClassifier', sklearn.neighbors.KNeighborsClassifier()))
# models.append(('AdaBoostClassifier', sklearn.ensemble.AdaBoostClassifier()))
# #models.append(('LogisticRegression', sklearn.linear_model.LogisticRegression()))

# results = []
# names = []
# seed = 7
# scoring = 'accuracy'
# for name, model in models:
#    kfold = model_selection.KFold(n_splits=10, random_state=seed, shuffle=(True))
#    cv_results = model_selection.cross_val_score(model, xtrain, ytrain, cv=kfold, scoring=scoring)
#    results.append(cv_results)
#    names.append(name)
#    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
#    print(msg)

# # boxplot algorithm comparison
# fig = plt.figure(figsize=(21,14))
# fig.suptitle('Algorithm Comparison')
# ax = fig.add_subplot(111)
# plt.boxplot(results)
# ax.set_xticklabels(names)
# plt.show()

#WE DETERMINED RANDOM FOREST IS THE BEST ALGORITHM
forest = sklearn.ensemble.RandomForestClassifier(n_estimators = 100)
forest.fit(xtrain,ytrain)
ypred = forest.predict(xtest)
from sklearn import metrics
print ("Accuracy of Random Forest is : ", metrics.accuracy_score(ytest, ypred))
forestcv = sklearn.model_selection.cross_validate(forest,xtrain,ytrain,cv=20)
print('20-fold average accuracy of RF is: ',forestcv['test_score'].mean())

n_estimators = [int(x) for x in np.linspace (100, 2000, 5)]
max_depth = [int(x) for x in np.linspace(1,30,5)]
min_samples_split = [int(x) for x in np.linspace(2,10,5)]
min_samples_leaf = [int(x) for x in np.linspace(1,10,5)]
hyperF = dict(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf)
#gridF = sklearn.model_selection.GridSearchCV(forest,hyperF,cv=3,verbose=3,n_jobs=-1)
#bestF = gridF.fit(xtrain,ytrain)
# print(n_estimators)

#BEST PARAMETERS IDENTIFIED AND INSERTED
forestOpt = sklearn.ensemble.RandomForestClassifier(random_state = 7, max_depth = 22, n_estimators = 1050, min_samples_split = 2, min_samples_leaf = 1)                            
modelOpt = forestOpt.fit(xtrain, ytrain)
ypr3d = modelOpt.predict(xtest)
print ("Accuracy of Tuned Random Forest is : ", metrics.accuracy_score(ytest, ypr3d))
forestOptcv = sklearn.model_selection.cross_validate(forestOpt,xtrain,ytrain,cv=20)
print('20-fold average accuracy of Tuned RF: ',forestOptcv['test_score'].mean())  

print("Execution time: ",datetime.datetime.now()-begin_time)