from fastapi import APIRouter, Query
from fastapi import Request
import pandas as pd

router = APIRouter()
@router.get("/dataset")
def get_dataset(request : Request, limit : int  = Query(default= None, ge = 0)):
    df = request.app.state.dataset
    if(limit is not None):
        df = df.head(limit)
    return df.to_dict(orient = "records")