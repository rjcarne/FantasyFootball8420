import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge




def predict_fantasy_output(X, Y, X_test, alpha):
    ridge = Ridge(alpha = alpha, normalize = True)
    ridge.fit(X, Y)

    pred = ridge.predict(X_test.reshape(1,-1))           
    
    return pred[0]

    