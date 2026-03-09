from fastapi import APIRouter, Query
import pandas as pd

PATH = "data_set/auto-mpg.csv"

router = APIRouter()
@router.get("/dataset")
def get_dataset(limit : int = Query(default = None, gt = 0)):
    df = pd.read_csv(PATH)
    if(limit):
        df = df.head(limit)
    return df.to_dict(orient="records")