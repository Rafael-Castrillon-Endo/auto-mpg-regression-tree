from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes import dataset
from contextlib import asynccontextmanager
from src.load_data import load_data
from api.routes import dataset
from api.routes import delete
from api.routes import train

@asynccontextmanager
async def lifespan(app : FastAPI):
    df = load_data()
    df["idx"] = df.index
    app.state.dataset = df
    yield

app = FastAPI(lifespan = lifespan)

origins = [
    "http://localhost:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(dataset.router)
app.include_router(delete.router)
app.include_router(train.router)

