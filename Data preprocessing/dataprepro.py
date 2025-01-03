# 1] Importing the libraries
import numpy as np
import matplotlib as plt
import pandas as pd
# 2]Importing the dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[ : , :-1].values
y = dataset.iloc[ : ,-1].values
print("the dependent and independant variables")
print(X)
print(y)
# 3]Taking care of missing data
from sklearn.impute import SimpleImputer
Imputer = SimpleImputer(missing_values=np.nan , strategy='mean')
Imputer.fit(X[ : ,1:3])
X[ : , 1:3] = Imputer.transform(X[ : , 1:3])
print("the updated independent variablr after takikng care of missing values")
print(X)
# 4] Data Encoding
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[0])],remainder='passthrough')
X=np.array(ct.fit_transform(X))
print("the encoded independent variable ")
print(X)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)
print("the dependent variable after encoding")
print(y)
# 5] Splitting the dataset into traning and testing set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=1)
print("the training independant variable")
print(X_train)
print("the traning dependent variable ")
print(X_test)
print("the testing independant variable")
print(X_train)
print("the testing dependent variable ")
print(X_test)
# 6] Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train[ : , 3: ]= sc.fit_transform(X_train[ : , 3: ])
X_test[ : , 3: ]=sc.fit_transform(X_test[: , 3:])
print("the scaled independant value")
print(X_train)
print(X_test)

