from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def knn_regression():
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('knn', KNeighborsRegressor(
            n_neighbors = 5,
            weights='distance',
            metric='minkowski',
            p=2
        ))
    ])
    return model