from fastapi import FastAPI

from api.routes import dataset


app = FastAPI()

app.include_router(dataset.router)

