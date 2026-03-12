from fastapi import APIRouter, Request, HTTPException
from src.train_model import train_model
from fastapi import Request
from sklearn.model_selection import train_test_split


router = APIRouter()
@router.post("/train")
def train(request : Request):
    df = request.app.state.dataset
    if(df is None):
        raise HTTPException(404, 'Dataset no cargado aún')
    X = df.drop(columns = ["mpg", "idx", "car name"], errors = 'ignore')
    y = df["mpg"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=42)
    request.app.state.X_test = X_test
    request.app.state.y_test = y_test
    model = train_model(X_train, y_train)
    request.app.state.model = model
    return {"message" : "modelo entrenado"}


