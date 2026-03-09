from sklearn.tree import DecisionTreeRegressor

def train_model(x, y):
    model = DecisionTreeRegressor()
    model.fit(x, y)
    return model