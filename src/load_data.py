import pandas as pd
from src.preprocess import preprocess

PATH = "./data_set/auto-mpg.csv"

def load_data():
    data = pd.read_csv(PATH)
    data = preprocess(data)
    return data
