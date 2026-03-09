from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

def show_tree(model, X):
    plt.figure(figsize=(20, 10))
    plot_tree(
        model,
        feature_names = X.columns,
        filled = True
    )
    plt.show()
