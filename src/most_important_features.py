import pandas as pd

def most_important_features(features_importance, X):
    importance = pd.Series(features_importance, index = X.columns)
    importance = importance.sort_values(ascending = False)
    return importance 