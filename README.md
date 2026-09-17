Customer Churn Prediction API

FastAPI REST API for predicting customer churn using XGBoost.

Features:

- Single customer prediction
- Batch prediction
- CSV upload prediction
- Interactive Swagger documentation

Tech Stack:

- Python
- FastAPI
- XGBoost
- Pandas
- Pydantic

API Endpoints:

GET /
POST /predict
POST /predict_batch
POST /predict_csv

Running Locally:

python -m venv venv

venv\Scripts\activate

pip install -r requirements.txt

uvicorn main:app --reload

API Documentation:

Once running:

http://127.0.0.1:8000/docs




### Data Attribution
The test data used in this repository is sourced from the [testcsv](https://www.kaggle.com/competitions/playground-series-s6e3/data) on Kaggle, created by [Kaggle](Kaggle.com). 

This data is licensed under the [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org) License. 
