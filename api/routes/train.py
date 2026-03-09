from fastapi import APIRouter
from api.schemas import TrainParams
from src.train_model import train_model

routes = APIRouter()
