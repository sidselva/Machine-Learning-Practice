#CODE POLYNOMIAL REGRESSION

#Import
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

x = np.array([5,15,25,35,45,55]).reshape((-1,1))
y = np.array([15,11,2,8,25,32])

transformer = PolynomialFeatures(degree=2, include_bias=False)
transformer.fit(x)

x_ = transformer.transform(x)

model = LinearRegression().fit(x_,y)
r_sq = model.score(x_, y)
print("R-squared value: ", round(r_sq,3))
print('intercept: ', round(model.intercept_,2))
print('coefficients: ', np.around(model.coef_,2))
y_pred = model.predict(x_)
print('predicted response: ', np.around(y_pred,2), sep='\n')

plt.plot(x,y,'o',color='blue');
plt.plot(x,y_pred,color = 'green',linewidth=1)

ML_line = mlines.Line2D([],[],color='green', label='ML prediction')
train = mlines.Line2D([],[],color='blue',marker='o',linestyle='None',label='Training data')
plt.legend(handles=[ML_line, train])
#ML_line = mlines.Line2D([],[],color='green', label='ML prediction')