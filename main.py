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
from src.KNN import knn_regression
from src.svm import svm_regression
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns 

import math

data = load_data()
data = data.drop(columns= ['car name'])
X = data.drop(columns= ['mpg'])
y = data['mpg']



def main_nw():
    model = neural_network()
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=42)
    
    model.fit(X_train, y_train)
    
    predict = model.predict(X_test)
    
    print("MSE Neural_Network -> ", mean_squared_error(y_test, predict))
    print("R2 Neural_Network ->: ", r2_score(y_test, predict))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    ax1.scatter(y_test, predict, alpha=0.5, color='blue')
    ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    ax1.set_xlabel('Valores Reales')
    ax1.set_ylabel('Predicciones')
    ax1.set_title('Predicción vs Realidad (Evaluación)')

    loss_values = model.named_steps['mlp'].loss_curve_
    ax2.plot(loss_values, color='red')
    ax2.set_xlabel('Iteraciones')
    ax2.set_ylabel('Pérdida (Loss)')
    ax2.set_title('Curva de Aprendizaje (Convergencia)')

    plt.tight_layout()
    plt.show()

    return model

def main_knn():
    model = knn_regression()
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size= 0.8, test_size= 0.2)

    model.fit(X_train, y_train)

    predict = model.predict(X_test)
    print("MSE KNN ->", mean_squared_error(y_test, predict))
    print("R2 KNN -> ",  r2_score(y_test, predict))
    return model

def main_svm():
    model = svm_regression()
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=42)
    
    model.fit(X_train, y_train)
    predict = model.predict(X_test)
    
    residuals = y_test - predict
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    ax1.scatter(y_test, predict, alpha=0.6, color='seagreen')
    ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    ax1.set_xlabel('Valores Reales')
    ax1.set_ylabel('Predicciones')
    ax1.set_title('SVR: Predicción vs Realidad')

    ax2.scatter(predict, residuals, alpha=0.6, color='darkorange')
    ax2.axhline(y=0, color='black', linestyle='--', lw=2)
    ax2.axhline(y=0.1, color='red', linestyle=':', label='Margen Epsilon ($\epsilon$)')
    ax2.axhline(y=-0.1, color='red', linestyle=':')
    ax2.set_xlabel('Valores Predichos')
    ax2.set_ylabel('Residuos (Error)')
    ax2.set_title('Análisis de Residuos (Margen $\epsilon$)')
    ax2.legend()

    plt.tight_layout()
    plt.show()

    print("MSE SVM -> ", mean_squared_error(y_test, predict))
    print("R2 SVM ->", r2_score(y_test, predict))
    return model

def main_s():
    model_nw = main_nw()
    model_knn = main_knn()
    model_svm = main_svm()

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