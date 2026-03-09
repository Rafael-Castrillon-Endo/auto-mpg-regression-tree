from sklearn.tree import DecisionTreeRegressor

def train_model(X, y):
    model = DecisionTreeRegressor(max_depth=4)
    model.fit(X, y)
    return model