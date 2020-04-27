#importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing Dataset
df = pd.read_csv('test1.csv')


#Creating Dependant and Independent variables
x = df.iloc[:,0:1].values
y = df.iloc[:,1].values



#Splitting Dataset into Train and test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

#Building LinearRegression Model
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(x_train,y_train)
y_pred = reg.predict(x_test)

#visualizing the Model
plt.scatter(x,y,color ='red')
plt.plot(x_train,reg.predict(x_train),color = 'blue')
plt.title('Linear Regression Model')
plt.xlabel('Area')
plt.ylabel('Prize')
plt.show()
