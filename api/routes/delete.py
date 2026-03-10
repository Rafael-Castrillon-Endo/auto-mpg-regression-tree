from fastapi import APIRouter, Request, HTTPException

router = APIRouter
@router.delete("/delete/{index}")
def delete_row(index : int , request : Request):
    df = request.app.state.dataset
    if(index not in df):
        raise HTTPException (status_code = 404, detail = "index invalido")
    df.drop(index = index, inplace = True)
    return {"message" : "Dato eliminado"}
