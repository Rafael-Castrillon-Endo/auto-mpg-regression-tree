from fastapi import APIRouter, Query
from src.load_data import load_data
import pandas as pd

router = APIRouter()
@router.get("/dataset")
def get_dataset(limit : int  = Query(default= None, gp = 0)):
    df = load_data()
    if(limit):
        df = df.head(limit)
    return df.to_dict(orient = "records")