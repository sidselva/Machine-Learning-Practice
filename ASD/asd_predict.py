#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 27 16:24:40 2020

Predicting the likelihood of ASD traits based on screening questions

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
import sklearn.naive_bayes

unk = 'A3'

begin_time = datetime.datetime.now()
df = pd.read_csv('data.csv',delimiter=(','))
df = df.drop(columns = ['Case_No','Ethnicity','Who completed the test','Qchat-10-Score'])
df['Sex'] = [0 if each == "f" else 1 for each in df['Sex']]
df['Jaundice'] = [0 if each == "no" else 1 for each in df['Jaundice']]
df['Family_mem_with_ASD'] = [0 if each == "no" else 1 for each in df['Family_mem_with_ASD']]
df['ASD'] = [0 if each == "No" else 1 for each in df['ASD']]
pd.set_option('display.max_columns', None)
print(df.describe())

y = df[unk]
x = df.drop(columns = [unk])

xtrain, xtest, ytrain, ytest = sklearn.model_selection.train_test_split(x,y,test_size=0.2)
corrMatrix = df.corr()
plt.figure(figsize=(10,10))
sns.heatmap(corrMatrix, annot=True)

models = []
models.append(('SupportVectorClassifier', sklearn.svm.SVC()))
models.append(('StochasticGradientDescentC', sklearn.linear_model.SGDClassifier()))
models.append(('RandomForestClassifier', sklearn.ensemble.RandomForestClassifier()))
models.append(('DecisionTreeClassifier', sklearn.tree.DecisionTreeClassifier()))
models.append(('GaussianNB', sklearn.naive_bayes.GaussianNB()))
models.append(('KNeighborsClassifier', sklearn.neighbors.KNeighborsClassifier()))
models.append(('AdaBoostClassifier', sklearn.ensemble.AdaBoostClassifier()))
# models.append(('LogisticRegression', sklearn.linear_model.LogisticRegression()))

results = []
names = []
seed = 7
scoring = 'accuracy'
for name, model in models:
    kfold = model_selection.KFold(n_splits=30, random_state=seed, shuffle=(True))
    cv_results = model_selection.cross_val_score(model, xtrain, ytrain, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
    
# boxplot algorithm comparison
fig = plt.figure(figsize=(21,14))
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()


# get a list of models to evaluate
def get_models():
	models = dict()
	# define number of trees to consider
	n_trees = np.arange(10,110,10).tolist()
	for n in n_trees:
		models[str(n)] = sklearn.ensemble.AdaBoostClassifier(n_estimators=n)
	return models
 
# evaluate a given model using cross-validation
def evaluate_model(model, x, y):
	# define the evaluation procedure
	cv = sklearn.model_selection.RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
	# evaluate the model and collect the results
	scores = sklearn.model_selection.cross_val_score(model, x, y, scoring='accuracy', cv=cv, n_jobs=-1)
	return scores
 
# get the models to evaluate
models = get_models()
# evaluate the models and store results
results, names = list(), list()
for name, model in models.items():
	# evaluate the model
	scores = evaluate_model(model, x, y)
	# store the results
	results.append(scores)
	names.append(name)
	# summarize the performance along the way
	print('>%s %.3f (%.3f)' % (name, np.mean(scores), np.std(scores)))
# plot model performance for comparison
plt.boxplot(results, labels=names, showmeans=True)
plt.show()

model = sklearn.ensemble.AdaBoostClassifier(n_estimators=30)
model.fit(xtrain,ytrain)
print(model.get_params())
ypred = model.predict(xtest)
print(ypred)
from sklearn import metrics
print ("Accuracy of AdaBoostClassifier is : ", metrics.accuracy_score(ytest, ypred))
modelcv = sklearn.model_selection.cross_validate(model,xtrain,ytrain,cv=40)
print('Multifold average accuracy of RF is: ',modelcv['test_score'].mean())

print("Execution time: ",datetime.datetime.now()-begin_time)
