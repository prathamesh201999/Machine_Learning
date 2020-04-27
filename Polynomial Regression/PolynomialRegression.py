#importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing Dataset
df  = pd.read_csv('Position_Salary.csv')



#Creating Dependant and Independent variables
x = df.iloc[:,1:2].values
y = df.iloc[:,2].values


from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(x,y)

#Building PolynomialRegression Model
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures( degree = 4)
x_poly = poly_reg.fit_transform(x,y)
poly_reg.fit(x_poly,y)
reg2 = LinearRegression()
reg2.fit(x_poly,y)


#visualizing the Model
x_grid = np.arange(min(x),max(x),0.1)
x_grid = x_grid.reshape(len(x_grid),1)
plt.scatter(x,y,color ='red')
plt.plot(x,reg2.predict(poly_reg.fit_transform(x)),color = 'blue')
plt.title('Polynomial Regression Model')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()
