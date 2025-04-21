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
Developed by: R Raihaan Ahmed
RegisterNumber:  212224040260
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
![image](https://github.com/user-attachments/assets/96ac83d1-9ef1-4df7-b802-ef0422b591eb)


### DATA INFO:
![image](https://github.com/user-attachments/assets/ee13ebd8-a0cf-443f-90ef-04e97e90b3cf)


### ISNULL() AND SUM():
![image](https://github.com/user-attachments/assets/d8081e47-49c2-4167-87e4-7fc5ac9aa6a2)


### DATA HEAD FOR SALARY:
![image](https://github.com/user-attachments/assets/8bc5407f-e1c5-46cc-85b3-2f219e8f391e)


### MEAN SQUARED ERROR:
![image](https://github.com/user-attachments/assets/b65894a6-d2f5-4961-a063-8437c3c2a641)


### R2 VALUE:
![image](https://github.com/user-attachments/assets/507fc1cc-ede7-4be9-9570-a97a686999a2)


### DATA PREDICTION:
![image](https://github.com/user-attachments/assets/ddc16c8a-2537-4a95-8532-112c392d44e7)




## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
