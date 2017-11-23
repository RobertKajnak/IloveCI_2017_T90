from simple_esn import SimpleESN
from sklearn.linear_model import Ridge
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from numpy import loadtxt
from pprint import pprint
from time import time
import numpy as np
from sklearn.model_selection import train_test_split


#X1 = loadtxt('aalborg.csv',  delimiter=",", skiprows=1)
#X2 = loadtxt('alpine-1.csv', delimiter=",", skiprows=1)
#X3 = loadtxt('f-speedway.csv', delimiter=",", skiprows=1)
#print("files read in")
#X12 = np.concatenate((X1, X2))
#data = np.concatenate((X12, X3))
data = loadtxt('threetracks.csv', delimiter=",", skiprows=1)

#X = atleast_2d(X).T
X = data[:, 3:]
y = data[:, :3]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
print("got data")
# Grid search
pipeline = Pipeline([('esn', SimpleESN(n_readout=3)),
                     ('ridge', Ridge(alpha = 0.01))])
parameters = {
    'esn__n_readout': [3],
    'esn__n_components': [1000,2000],
    'esn__weight_scaling': [0.9, 1.25],
    'esn__damping': [0.3, 0.5, 0.8],
    'ridge__alpha': [0.05, 0.01, 0.001]
}
grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1, cv=3)
print("grid initialized")
print ("Starting grid search with parameters")
pprint (parameters)
t0 = time()
grid_search.fit(X_train, y_train)
print ("done in %0.3f s" % (time()-t0))

print ("Best score on training is: %0.3f" % grid_search.best_score_)
best_parameters = grid_search.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
    print ("\t%s: %r" % (param_name, best_parameters[param_name]))

y_true, y_pred = y_test, grid_search.predict(X_test)
err = mean_squared_error(y_true, y_pred)


#Best score on training is: 0.011
#	esn__damping: 0.8
#	esn__n_components: 2000
#	esn__n_readout: 3
#	esn__weight_scaling: 1.25
#	ridge__alpha: 0.01

