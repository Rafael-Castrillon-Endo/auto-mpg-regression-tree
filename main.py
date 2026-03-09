from src.load_data import load_data
from src.preprocess import preprocess
from src.train_model import train_model
from sklearn.model_selection import train_test_split

data_clean = preprocess(load_data())

data_clean = preprocess(load_data())

x = data_clean.mpg("mpg", axis = 1)
y = data_clean["mpg"]

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size= 0.8, test_size= 0.2, random_state = 1)

model = train_model(x_train, y_train)
