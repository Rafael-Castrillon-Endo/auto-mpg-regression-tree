from sklearn.ensemble import RandomForestRegressor

def train_model_forest(X, y, n_trees):
    model = RandomForestRegressor(n_estimators= 5, max_depth=3,  min_samples_leaf= 10, min_samples_split = 30)
    model.fit(X, y)
    return model