from simple_esn import SimpleESN
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor
from numpy import loadtxt, atleast_2d
import matplotlib.pyplot as plt
import numpy as np
from sklearn.externals import joblib

X1 = loadtxt('aalborg.csv',  delimiter=",", skiprows=1)
T = loadtxt('alpine-1.csv', delimiter=",", skiprows=1)
X2 = loadtxt('f-speedway.csv', delimiter=",", skiprows=1)
X = np.concatenate((X1, X2))

#X = atleast_2d(X).T

X_train = X[:, 3:]
y_train = X[:, :3]
X_test = T[:, 3:]
y_test = T[:, :3]

#X_test = X[train_length:train_length+test_length]
#y_test = X[train_length+1:train_length+test_length+1]

# Simple training
my_esn = SimpleESN(n_readout=3, n_components=1000,
               damping = 0.3, weight_scaling = 0.9)
echo_train = my_esn.fit_transform(X_train)
regr = Ridge(alpha = 0.01)

regr.fit(echo_train, y_train)
echo_test = my_esn.transform(X_test)
y_true, y_pred = y_test, regr.predict(echo_test)
err = mean_squared_error(y_true, y_pred)
print(y_true, y_pred, err)


joblib.dump(my_esn, 'ESN0.pkl') 
joblib.dump(regr, 'regr.pkl') 
