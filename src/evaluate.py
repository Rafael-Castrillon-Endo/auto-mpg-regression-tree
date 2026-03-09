from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

def evalute_mean_squared_error(predictions, y_test):
    e_mse = mean_squared_error(y_test, predictions)
    return e_mse

def evalute_r2_score(predictions, y_test):
    e_r2_score =  r2_score(y_test, predictions)
    return e_r2_score