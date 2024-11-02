from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import pickle
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.responses import JSONResponse
from fastapi import BackgroundTasks
from fastapi.responses import FileResponse
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React app's origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#load clustering model and preprocessor
with open("Saved_Models/kmeans_model.pkl", "rb") as f:
    kmeans_model = pickle.load(f)

with open("Saved_Models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("Saved_Models/imputer.pkl", "rb") as f:
    imputer = pickle.load(f)

# Load the trained model and scaler using pickle
with open("Saved_Models/model_better_1.pkl", "rb") as f:
    model = pickle.load(f)

with open("Saved_Models/scaler-regression.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("Saved_Models/X_columns.pkl", "rb") as f:
    X_columns = pickle.load(f)

# Define the structure for input data
class HouseData(BaseModel):
    Suburb: str
    Rooms: int
    Date: str
    Distance: float
    Postcode: int
    Bedroom2: int
    Bathroom: int
    Landsize: float
    BuildingArea: float
    YearBuilt: int
    CouncilArea: str
    Regionname: str
    Propertycount: int

class ClusterData(BaseModel):
    Rooms: int
    Price: float
    Distance: float
    Bathroom: int
    Car: int
    Landsize: float
    BuildingArea: float
    YearBuilt: int
    Lattitude: float
    Longtitude: float

@app.post("/predict")
async def predict_price(house_data: HouseData):
    # Convert incoming data into a DataFrame
    data = {
        'Suburb': [house_data.Suburb],
        'Rooms': [house_data.Rooms],
        'Date': [house_data.Date],
        'Distance': [house_data.Distance],
        'Postcode': [house_data.Postcode],
        'Bedroom2': [house_data.Bedroom2],
        'Bathroom': [house_data.Bathroom],
        'Landsize': [house_data.Landsize],
        'BuildingArea': [house_data.BuildingArea],
        'YearBuilt': [house_data.YearBuilt],
        'CouncilArea': [house_data.CouncilArea],
        'Regionname': [house_data.Regionname],
        'Propertycount': [house_data.Propertycount]
    }

    new_data_df = pd.DataFrame(data)
    
    new_data_df = pd.get_dummies(new_data_df, columns=['Suburb', 'Date', 'CouncilArea', 'Regionname'])

    missing_cols = list(set(X_columns) - set(new_data_df.columns))
    new_data_df = pd.concat([new_data_df, pd.DataFrame(0, index=new_data_df.index, columns=missing_cols)], axis=1)

    new_data_df = new_data_df[X_columns]

    new_data_scaled = scaler.transform(new_data_df)

    predicted_price = model.predict(new_data_scaled)

    return {"predicted_price": predicted_price[0]}

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail, "error": "An error occurred"}
    )

@app.get("/get-csv")
async def get_csv():
    file_path = os.path.join("database", "csv2.csv")
    return FileResponse(file_path, media_type="text/csv", filename="csv2.csv")

# Define the endpoint for clustering
@app.post("/predict-cluster")
async def predict_cluster(cluster_data: ClusterData):
    input_df = pd.DataFrame([cluster_data.model_dump()])

    input_imputed = imputer.transform(input_df)
    input_scaled = scaler.transform(input_imputed)

    cluster = kmeans_model.predict(input_scaled)
    return {"cluster": int(cluster[0])}
