## PERFORM PREDICTIONS ON RED WINE QUALITY
## DATASET FROM http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
import statsmodels.formula.api as smf
import statsmodels.api as sm
import sklearn
from sklearn import model_selection, preprocessing, feature_selection, ensemble, linear_model, metrics, decomposition
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from lime import lime_tabular
import datetime

begin_time = datetime.datetime.now()

csv = pd.read_csv("winequality-red.csv", delimiter=';')
#print("read successfully")
#print(csv.head(2))
#print(csv.keys())

#SEPARATE INDEPENDENT FROM RESPONSE VARIABLE
x = csv.drop('quality',axis=1)
#print (x)
y = csv['quality']
#print (y)

#SPLIT TEST AND TRAINING DATA
xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size=0.15)

#RESCALE TEST AND TRAINING DATA
#Scale this down to a smaller range to avoid biasing
sc = StandardScaler()
xtrain = sc.fit_transform(xtrain)
xtest = sc.fit_transform(xtest)

#EXAMINE DATA
#This will show mean, min, max, std dev, & quartiles
print(csv.describe())

#HISTOGRAM
csv.hist(bins=10, figsize=(10,10))
plt.show()

#DENSITY
csv.plot(kind='density', subplots = 'True', layout=(4,3), figsize=(10,10), sharex=False)
plt.show()

#CREATE PIVOT TABLE
columns = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 
           'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
           'pH', 'sulphates', 'alcohol']
csv_pivot_table = csv.pivot_table(columns,['quality'],aggfunc='median')
print(csv_pivot_table)

#CREATE CORRELATION MATRIX
corr_matrix = csv.corr()
print(corr_matrix["quality"].sort_values(ascending=False))
fig, ax = plt.subplots(figsize=(10,10))
colormap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr_matrix,cmap=colormap,annot=True, fmt=".2f")
plt.show()

#SCATTERPLOT MATRIX
sm = pd.plotting.scatter_matrix(csv, figsize=(12,12), diagonal='kde')
[s.xaxis.label.set_rotation(40) for s in sm.reshape(-1)]
[s.yaxis.label.set_rotation(0) for s in sm.reshape(-1)]
[s.get_yaxis().set_label_coords(-0.6,0.5) for s in sm.reshape(-1)]
[s.set_xticks(()) for s in sm.reshape(-1)]
[s.set_yticks(()) for s in sm.reshape(-1)]
plt.show()

#BIN OUTPUTS
bins = (2,6,8)
group_names = ['bad', 'good']
csv['quality']=pd.cut(csv['quality'], bins=bins, labels=group_names)
label_quality = LabelEncoder()
csv['quality'] = label_quality.fit_transform(csv['quality'])
print(csv['quality'].value_counts())

sns.countplot(csv['quality'])
plt.show()

#EVALUATE ALGORITHMS
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

#OPTIMIZE ALGORITHM PARAMETERS (HYPERPARAMETERS)
# def svc_param_selection(x,y,nfolds):
#     param = {
#         'C': [0.1, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4],
#         'kernel': ['linear', 'rbf'],
#         'gamma': [0.1, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4]}
#     grid_search = sklearn.model_selection.GridSearchCV(sklearn.svm.SVC(), param_grid=param, scoring='accuracy', cv=nfolds)
#     grid_search.fit(x,y)
#     return grid_search.best_params_
# print(svc_param_selection(xtrain, ytrain,10))

svc = sklearn.svm.SVC(C = 1.3, gamma =  1.3, kernel= 'rbf')
svc.fit(xtrain, ytrain)
pred_svc = svc.predict(xtest)
print('Confusion matrix')
print(sklearn.metrics.confusion_matrix(ytest, pred_svc))
print('Classification report')
print(sklearn.metrics.classification_report(ytest, pred_svc))
print('Accuracy score',sklearn.metrics.accuracy_score(ytest, pred_svc))

print(datetime.datetime.now()-begin_time)
