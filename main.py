from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from src.load_data import load_data
from src.train_model import train_model
from src.evaluate import evalute_mean_squared_error, evalute_r2_score
from src.train_model_forest import train_model_forest
from src.most_important_features import most_important_features
from src.show_tree import show_tree
from src.graphs import plot_model_year_distribution
from src.graphs import plot_model_cylinders
from src.graphs import plot_mpg_weight_regression
from src.graphs import plot_mpg_vs_displacement
from src.graphs import plot_density_comparison
from src. graphs import plot_feature_importance
from src.neural_network import neural_network


import math

data = load_data()
data = data.drop(columns= ['car name'])
X = data.drop(columns= ['mpg'])
y = data['mpg']



def main_nw():
    model = neural_network()
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size= 0.8, test_size= 0.2)
    model.fit(X_train, y_train)
    predict = model.predict(X_test)
    print("MSE: ", mean_squared_error(y_test, predict))
    print("R2: ", r2_score(y_test, predict))
    return model

def main_tree_s():
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size= 0.8, test_size= 0.2)
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)

    mse = evalute_mean_squared_error(y_test, predictions)
    rrscore = evalute_r2_score(y_test, predictions)

    print(f"mean squared error -> {mse}")
    print(f"r2 score -> {rrscore}")
    print(f" % de error -> {math.sqrt(mse)}")
    show_tree(model, X)
    print("=====================================================")
    importance = model.feature_importances_

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
        arbol_i = model.estimators_[i]
        #show_tree(arbol_i, X)
        predict = arbol_i.predict(X_test)
        plot_density_comparison(y_test, predict)
        #plot_feature_importance(arbol_i.feature_importances_, X)
    print("=====================================================")


main_nw()
#main_tree_frst()
#plot_model_cylinders(data)
#plot_mpg_weight_regression(data)
#plot_mpg_vs_displacement(data)
#main_tree_s()