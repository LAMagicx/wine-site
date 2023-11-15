from fastapi import FastAPI, Request, Form, Response, Depends, status, APIRouter
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.exceptions import HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from typing import Annotated, Union
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

## Functions

## Routes

@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("landing.html", {"request": request})
