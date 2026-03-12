from sklearn.model_selection import train_test_split
from src.load_data import load_data
from src.train_model import train_model
from src.evaluate import evalute_mean_squared_error, evalute_r2_score
from src.train_model_forest import train_model_forest
from src.most_important_features import most_important_features
import math

data = load_data()
data = data.drop(columns= ['car name'])
X = data.drop(columns= ['mpg'])
y = data['mpg']

def main_tree_s():
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size= 0.8, test_size= 0.2, random_state= 42)
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)

    mse = evalute_mean_squared_error(y_test, predictions)
    rrscore = evalute_r2_score(y_test, predictions)

    print(f"mean squared error -> {mse}")
    print(f"r2 score -> {rrscore}")
    print(f" % de error -> {math.sqrt(mse)}")
    print("=====================================================")

def main_tree_frst():
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size= 0.8, test_size= 0.2, random_state= 42)
   
    n_trees = 5

    model = train_model_forest(X_train, y_train, n_trees)
    trees = model.estimators_
    predictions = model.predict(X_test)

    mse = evalute_mean_squared_error(predictions, y_test)
    rrscore = evalute_r2_score(predictions, y_test)
    print(f"mean_squared_error -> {mse}")
    print(f"r2 score -> {rrscore}")
    print(f"% de error -> {math.sqrt(mse)}")
    print("=====================================================")

    for i in range(n_trees):
        print(f"Random Forest Arbol {i + 1}")
        featur_importance = model.estimators_[i].feature_importances_

        ft_importance = most_important_features(featur_importance, X)
        print(" == importance features == ")
        print(ft_importance)
        print(" == Depth == ")
        print(model.estimators_[i].get_depth())

    print("=====================================================")



main_tree_frst()