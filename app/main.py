from fastapi import FastAPI, Request, Form, Response, Depends, status, APIRouter
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.exceptions import HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from typing import Annotated, Union, Dict, Optional, List
from pydantic import BaseModel
from pymongo import MongoClient
from uuid import uuid4, UUID
from datetime import datetime, timedelta
import random, json
from model import Model
import pandas as pd

## Settings

app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static", html=True), name="static")

# MongoDB connection
# mongo_client = MongoClient("mongodb://root:pass@mongo:27017/?authMechanism=DEFAULT")

## Classes

class Wine(BaseModel):
    fixed_acidity: float
    volatile_acidity: float
    citric_acid : float
    residual_sugar : float
    chlorides : float
    free_sulfur_dioxide : int
    total_sulfur_dioxide : int
    density : float
    ph : float
    sulphates : float
    alcohol : float
    quality : Optional[float] = None


MODEL = Model("model/data.csv")

## Routes

@app.post("/api/predict")
def predict_grade(wine: Wine) -> List[float]:
    return MODEL.predict(wine)

@app.get("/api/predict")
def get_perfect_wine() :
    return Wine  # In type WINE

@app.get("/api/model")
def get_serialized():
    model_json = MODEL.to_json()
    return model_json

@app.get("/api/model/description")
def get_model_info():
    return "Kill me"

@app.put("/api/model")
def add_entry(wine: Wine):
    s = ','.join([str(x) for x in list(json.loads(wine.json()).values())])
    lines = open("model/data.csv", "r").readlines()
    print(lines[-1].split(",")[-1])
    last_id = int(lines[-1].split(",")[-1])
    lines.append(s + "," + str(last_id + 1) + "\n")
    open("model/data.csv", "w").writelines(lines)

@app.post("/api/model/retrain")
def retrain_model():
    MODEL.train("model/data.csv")
