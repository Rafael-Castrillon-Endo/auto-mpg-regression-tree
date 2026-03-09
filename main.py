from src.load_data import load_data
from src.preprocess import preprocess
from src.train_model import train_model
from src.evaluate import evalute_mean_squared_error
from src.evaluate import evalute_r2_score
from sklearn.model_selection import train_test_split

data_clean = preprocess(load_data())

data_clean = preprocess(load_data())

x = data_clean.mpg("mpg", axis = 1)
y = data_clean["mpg"]

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size= 0.8, test_size= 0.2, random_state = 1)

model = train_model(x_train, y_train)

predictions = model.predict(x_test)

print(f" Model mean squared error -> {evalute_mean_squared_error(predictions, y_test)}")
print(f" Model r2 score -> {evalute_r2_score(predictions, y_test)}")
