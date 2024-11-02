import json
import requests

url = 'https://http://127.0.0.1:8000/predict'

input_data = {
    "Suburb": "Abbotsford",
    "Rooms": 3,
    "Date": "2016-03-12",
    "Distance": 2.5,
    "Postcode": 3067,
    "Bedroom2": 3,
    "Bathroom": 2,
    "Landsize": 126,
    "BuildingArea": 109,
    "YearBuilt": 1990,
    "CouncilArea": "Yarra",
    "Regionname": "Northern Metropolitan",
    "Propertycount": 4019
}

input_data = json.dumps(input_data)

response = requests.post(url, json==input_data)
print(response.json())