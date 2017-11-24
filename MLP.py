# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 23:37:55 2017

@author: student
"""

from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor
from numpy import loadtxt, atleast_2d
import matplotlib.pyplot as plt
import numpy as np
from sklearn.externals import joblib
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
#from sknn.mlp import Classifier, Layer

from sklearn.preprocessing import StandardScaler


#from sklearn.pipeline import Pipeline
#from sklearn.preprocessing import MinMaxScaler

# check https://scikit-neuralnetwork.readthedocs.io/en/latest/guide_sklearn.html for documentation

X1 = loadtxt('aalborg.csv',  delimiter=",", skiprows=1)
T = loadtxt('alpine-1.csv', delimiter=",", skiprows=1)
X2 = loadtxt('f-speedway.csv', delimiter=",", skiprows=1)
data = np.concatenate((X1, X2, T))



X = data[:, 3:]
y = data[:, :3]

#splitting up data into train and test
X_train, X_test, y_train, y_test = train_test_split(X,y)

#scale the data

scaler = StandardScaler()
scaler.fit(X_train)

# Now apply the transformations to the data:
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

regr = Ridge(alpha = 0.01)

# train the Mlp
mlp = MLPRegressor(hidden_layer_sizes=(30,30,30))
mlp.fit(X_train,y_train)
y_true, y_pred = y_test, mlp.predict(X_test)
err = mean_squared_error(y_true, y_pred)
print(y_true, y_pred, err)


import pickle
pickle.dump(mlp, open('mlp.pkl', 'wb'))
