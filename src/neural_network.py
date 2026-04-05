from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def neural_network():
    model = Pipeline([
        ('Scaler', StandardScaler()),
        ('mlp', MLPRegressor(
            hidden_layer_sizes = (64, 32),
            activation = 'relu',
            solver = 'adam',
            max_iter = 500,
            random_state = 42  
        ))
    ])
    return model
