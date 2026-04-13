import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

dataset = pd.read_csv("D:\Hardik\Machine-learning-programs-main\Machine-learning-programs-main\Datasets\expvssal.csv")

x=dataset.iloc[:,0:1]
y=dataset.iloc[:,1]

print(x)
print(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

#creating and fitting the model
model = LinearRegression()
model.fit(x_train,y_train)

#Making Prediction
y_pred_test = model.predict(x_test)
y_pred_train = model.predict(x_train)

y_pred=model.predict([[11]])

print("Predicted Salary is:", y_pred)

#plotting results

plt.scatter(x_train, y_train, color='lightcoral')
plt.plot(x_train, y_pred_train, color = 'firebrick')
plt.title('Exprience vs Salary')
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.legend(['x_train/pred(y_test)', 'x_train/y_train'], title='exp/sal', loc='best', facecolor='white')
plt.box(False)
plt.show()
