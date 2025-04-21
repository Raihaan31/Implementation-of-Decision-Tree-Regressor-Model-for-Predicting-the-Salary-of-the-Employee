# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the libraries and read the data frame using pandas.
2. Calculate the null values present in the dataset and apply label encoder.
3. Determine test and training data set and apply decison tree regression in dataset.
4. Calculate Mean square error,data prediction and r2.

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: 
RegisterNumber:  
*/
```
```py
import pandas as pd
data=pd.read_csv("Salary.csv")
data.head()

data.info

data.isnull().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["Position"]=le.fit_transform(data["Position"])
data.head()

x=data[["Position","Level"]]
y=data[["Salary"]]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2,random_state=2)

from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

from sklearn import metrics
mse=metrics.mean_squared_error(y_test, y_pred)
mse

r2=metrics.r2_score(y_test,y_pred)
r2

dt.predict([[5,6]])
```
## Output:

### DATA HEAD:
![image](https://github.com/user-attachments/assets/522c6ed0-dc02-4cdd-b8bd-db3ba241d946)


### DATA INFO:
![image](https://github.com/user-attachments/assets/f710cdd1-e02d-474c-9bcb-1b06e1703f4f)


### ISNULL() AND SUM():
![image](https://github.com/user-attachments/assets/b06c86a6-8d8c-43b0-b6e6-6cf25d3dedb0)


### DATA HEAD FOR SALARY:
![image](https://github.com/user-attachments/assets/25a9d982-f39d-4bc7-8e70-b86de6571e04)


### MEAN SQUARED ERROR:
![image](https://github.com/user-attachments/assets/ab74598b-2ffe-4cad-a8fc-a615057c1744)


### R2 VALUE:
![image](https://github.com/user-attachments/assets/ab74598b-2ffe-4cad-a8fc-a615057c1744)


### DATA PREDICTION:
![image](https://github.com/user-attachments/assets/ce6b76c4-13eb-406a-a61e-56b9f6cd66a1)


## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
