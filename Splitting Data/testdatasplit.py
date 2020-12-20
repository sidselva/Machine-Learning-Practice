#CODE & PLOT A STANDARD LINEAR REGRESSION

#Import data
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

#Arrange datasets into testing and training data
x = np.arange(20).reshape(-1, 1)
y = np.array([5, 12, 11, 19, 30, 29, 23, 40, 51, 54, 74, 62, 68, 73, 89, 84, 89, 101, 99, 106])
xtrain, xtest, ytrain, ytest = train_test_split(x,y,train_size=0.25)

#Implement regression & prediction model
model = LinearRegression().fit(xtrain,ytrain)
ypredict = model.predict(xtest)

#Confirm accuracy
accuracy = model.score(xtest,ytest)
textstr = "R-squared value: " + str(round(accuracy,3))

#Plot data
plt.plot(xtrain,ytrain,'o',color='blue');
plt.plot(xtest,ytest,'o',color='black');
plt.plot(xtest,ypredict,color = 'green',linewidth=1)
ML_line = mlines.Line2D([],[],color='green', label='ML prediction')
train = mlines.Line2D([],[],color='blue',marker='o',linestyle='None',label='Training data')
test = mlines.Line2D([],[],color='black',marker='o',linestyle='None',label='Test data')
plt.legend(handles=[train,test,ML_line])
plt.text(10, 5, textstr, fontsize=10)