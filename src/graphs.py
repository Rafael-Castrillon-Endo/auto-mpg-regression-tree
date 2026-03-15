import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_model_year_distribution(df):
    plt.figure(figsize=(10,6))
    
    sns.countplot(x="model year", data=df)

    plt.title("Distribution of Cars by Model Year")
    plt.xlabel("Model Year")
    plt.ylabel("Number of Cars")
    plt.show()

def plot_model_cylinders(df):
    plt.figure(figsize=(10,6))
    
    sns.countplot(x="cylinders", data=df)

    plt.title("Distribution of cylinders")
    plt.xlabel("cylinders")
    plt.ylabel("Number of Cars")
    plt.show()

def plot_mpg_weight_regression(df):
    plt.figure(figsize=(8,6))
    sns.regplot(x="weight", y="mpg", data=df, scatter_kws={"alpha":0.6})
    plt.title("MPG vs Weight")
    plt.xlabel("Weight")
    plt.ylabel("MPG")
    plt.show()

def plot_mpg_vs_displacement(df):

    plt.figure(figsize=(8,6))

    sns.regplot(
        x="displacement",
        y="mpg",
        data=df,
        scatter_kws={"alpha":0.6}
    )

    plt.title("MPG vs Displacement")
    plt.xlabel("Engine Displacement")
    plt.ylabel("MPG")

    plt.show()

def plot_density_comparison(y_test, predictions):
    plt.figure(figsize=(10, 6))
    
    sns.kdeplot(y_test, 
                fill=True, 
                color="skyblue", 
                label="Valores Reales", 
                alpha=0.5, 
                linewidth=2)
    
    sns.kdeplot(predictions, 
                fill=True, 
                color="orange", 
                label="Predicciones", 
                alpha=0.5, 
                linewidth=2)
    
    plt.title('Comparación de Densidad: Real vs. Predicho', fontsize=14)
    plt.xlabel('Valor de la Variable')
    plt.ylabel('Densidad')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.show()

def plot_feature_importance(features_importance, X):

    importance = pd.Series(features_importance, index=X.columns)
    importance = importance.sort_values(ascending=False)

    plt.figure(figsize=(8,6))

    sns.barplot(
        x=importance.values,
        y=importance.index
    )

    plt.title("Feature Importance (Decision Tree)")
    plt.xlabel("Importance")
    plt.ylabel("Features")

    plt.show()