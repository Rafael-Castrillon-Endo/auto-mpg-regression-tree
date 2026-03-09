import pandas as pd

PATH = "./data_set/auto-mpg.csv"

def load_data():
    data = pd.read_csv(PATH)
    return data
