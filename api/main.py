from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes import dataset
from contextlib import asynccontextmanager
from src.load_data import load_data
from api.routes import dataset

@asynccontextmanager
async def lifespan(app : FastAPI):
    app.state.dataset = load_data()
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

