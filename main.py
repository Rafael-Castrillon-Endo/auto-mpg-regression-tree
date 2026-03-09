from src.load_data import load_data
from src.preprocess import preprocess
from src.train_model import train_model
from src.evaluate import evalute_r2_score
from src.show_tree import show_tree
from src.evaluate import evalute_mean_squared_error
from sklearn.tree import plot_tree


from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

data_clean = preprocess(load_data())
data_clean = data_clean.drop("car name", axis = 1)

X = data_clean.drop("mpg", axis = 1)
y = data_clean["mpg"]

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size= 0.8, test_size= 0.2, random_state = 1)

model = train_model(X_train, y_train)

predictions = model.predict(X_test)

print(f" Model mean squared error -> {evalute_mean_squared_error(predictions, y_test)}")
print(f" Model r2 score -> {evalute_r2_score(predictions, y_test)}")

show_tree(model, X)
