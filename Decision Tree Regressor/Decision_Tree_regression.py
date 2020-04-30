import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



df = pd.read_csv('Position_Salary.csv')



x = df.iloc[:,1:2].values
y = df.iloc[:,2].values




from sklearn.tree import DecisionTreeRegressor
reg = DecisionTreeRegressor(random_state = 0)
reg.fit(x,y)


x_grid = np.arange(min(x),max(x),0.01)
x_grid = x_grid.reshape(len(x_grid),1)
plt.scatter(x,y,color = 'red')
plt.plot(x_grid,reg.predict(x_grid),color = 'blue')
plt.title('Decision Tree Regressor')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()
