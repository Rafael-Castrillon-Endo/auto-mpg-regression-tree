from sklearn.tree import DecisionTreeRegressor

def train_model(X, y):
    model = DecisionTreeRegressor()
    model.fit(X, y)
    return model