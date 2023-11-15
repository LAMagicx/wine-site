from fastapi import FastAPI, Request, Form, Response, Depends, status, APIRouter
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.exceptions import HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from typing import Annotated, Union, Dict
from pydantic import BaseModel
from pymongo import MongoClient
from uuid import uuid4, UUID
from datetime import datetime, timedelta
import random

## Settings

app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static", html=True), name="static")

# MongoDB connection
mongo_client = MongoClient("mongodb://root:pass@mongo:27017/?authMechanism=DEFAULT")

## Classes

class Wine(BaseModel):
    fixed_acidity:float
    volatile_acidity:float
    citric_acid : float
    residual_sugar : float
    chlorides : float
    free_sulfur_dioxide : int
    total_sulfur_dioxide : int
    density : float
    ph : float
    sulphates : float
    alcohol : float
    quality : int
    id : int
    
class Model(BaseModel):
    parameters : str
    metrics : str
    others : str

## Functions

## Routes

@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("landing.html", {"request": request})

@app.post("/api/predict")
def predict_grade(wine: Wine):
    return 0 #GRADE PREDICTED

@app.get("/api/predict")
def get_perfect_wine() :
    return Wine #In type WINE

@app.get("/api/model")
def get_serialized():
    return model #Euhh idk

@app.get("/api/model/description")
def get_model_info():
    return info

@app.put("/api/model")
def add_entry(wine: Wine):
    #Add the wine to the catalog
    wine
    
@app.post("/api/model/retrain")
def retrain_model():
    doTheThingWithModelPy