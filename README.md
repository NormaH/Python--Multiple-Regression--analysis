# Python--Multiple-Regression--analysis
Regression model with a categorical variable &amp; the rest are measurable independent variables; then using Step wise regression for optimum regressor selection 

Created on Sun Oct 15 16:51:15 2017

@author: t

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

path =  "rC:\\Users\\t\\.spyder-py3\\Python-Data-Science-and-Machine-Learning\50_Startups.csv " 
 
dataset= pd.read_csv(r"C:\Users\t\.spyder-py3\Python-Data-Science-and-Machine-Learning\50_Startups.csv", low_memory=False)

##Setting pandas to show all columns in dataframes
pandas.set_option('display.max_columns', None)

# bug fix for display formats to avoid run time errors
pd.set_option('display.float_format', lambda x: '%f' %x)

# Setting variables to work as numeric
dataset['R&D Spend'] = dataset['R&D Spend'].convert_objects(convert_numeric=True)
dataset['Administration'] = dataset['Administration'].convert_objects(convert_numeric=True)
dataset['Marketing Spend'] = dataset['Marketing Spend'].convert_objects(convert_numeric=True) 

# Data Preprocessing

# Importing the dataset
dataset = pd.read_csv(r'C:\Users\t\.spyder-py3\Python-Data-Science-and-Machine-Learning\50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

# Encoding categorical data
# Encoding the Independent Variable

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

#  Remove this from the transferred code (Encoding the Dependent Variable) as it does not impact Y
###   labelencoder_y = LabelEncoder()
###  y = labelencoder_y.fit_transform(y)

#Avoiding the Dummy Variable trap  (taking away California), as some libraries prompt to do it manually
X = X[:, 1:]

#Spliting the ds in thr Training and Test ds
 
from  sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size =0.2, random_state =0)
 
 # Feature Scaling
### sklearn.preprossing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform)X_train)
X_test = sc.X.transform)X_test)

## Feature scaling 
from sklearn.preprocessing import StandardScalar
sc_X = StandardScalar()
X_train = sc.X.fit_transform(X_train)
X_test =sc_X.transform(X_test)

# Fitting ML Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predictor the Test set results
y_pred = regressor.predict(X_test)

# Building the optimal model using Backward Elimination (need to add a column of 1 for B0, therefore using 1)
import statsmodel.formula.api as sm
## this is as we are adding, next is to put the 1s @ initial: X =np.append(arr =X, values =np.ones((50, 1)).astype(int), axis=1)
## X =np.append(arr =X, values =np.ones((50, 1)).astype(int), axis=1)
X= np.append(arr = np.ones((50,1)).astype(int), values= X, axis=1)

#Building the optimal model using Backward Elimination
import statsmodels.formula.api as sm
X =np.append(arr =np.ones((50,1)).astype(int), values =X,  axis=1)
X_opt = X [:, [0,1, 2,3,4,5]]
# specifies the dependent var and endog, and exog is array like: A knob x k array , nbos is # of obs, and k # of regressors)
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

## Removing variable 3 or index 2
X_opt = X [:, [0,1, 3,4,5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X [:, [0, 3]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
