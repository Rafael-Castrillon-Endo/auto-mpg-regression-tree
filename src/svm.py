from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def svm_regression():
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVR(
            kernel='rbf',
            C =1.0,
            epsilon=0.1,
            gamma='scale'
        ))
    ])
    return model