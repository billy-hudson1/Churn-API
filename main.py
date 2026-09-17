from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import joblib
import pandas as pd
import xgboost as xgb
from fastapi import UploadFile, File
import io


model = xgb.XGBClassifier()
model.load_model("xgboost_model.json")

app = FastAPI()

@app.post("/predict_csv")
async def predict_csv(file: UploadFile = File(...)):
    contents = await file.read()
    raw_df = pd.read_csv(io.BytesIO(contents))

    # Convert each row into a Customer object, so it goes through
    # the exact same validation as the other endpoints
    customers = []
    for _, row in raw_df.iterrows():
        customer_dict = row.to_dict()
        customer_dict.pop("id", None)  # drop id if present, model doesn't use it
        customers.append(Customer(**customer_dict))

    df = encode_customers(customers)
    probs = model.predict_proba(df)[:, 1]

    results = []
    for i, p in enumerate(probs):
        row_id = int(raw_df.iloc[i]["id"]) if "id" in raw_df.columns else i
        results.append({"id": row_id, "churn_probability": round(float(p), 4)})

    return results

model_columns = joblib.load("model_columns.pkl")

class Customer(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float


def encode_customers(customers: List[Customer]) -> pd.DataFrame:
    """Turn a list of Customer objects into a model-ready DataFrame,
    with exactly one row per customer, encoded and column-aligned."""
    df = pd.DataFrame([c.dict() for c in customers])

    df["gender"] = df["gender"].map({"Female": 0, "Male": 1})
    df["Partner"] = df["Partner"].map({"No": 0, "Yes": 1})
    df["Dependents"] = df["Dependents"].map({"No": 0, "Yes": 1})
    df["PhoneService"] = df["PhoneService"].map({"No": 0, "Yes": 1})
    df["MultipleLines"] = df["MultipleLines"].map({"No phone service": 0, "No": 0, "Yes": 1})

    df["DSL"] = (df["InternetService"] == "DSL").astype(int)
    df["FiberOptic"] = (df["InternetService"] == "Fiber optic").astype(int)
    df.drop(columns=["InternetService"], inplace=True)

    df["OnlineSecurity"] = (df["OnlineSecurity"] == "Yes").astype(int)
    df["OnlineBackup"] = (df["OnlineBackup"] == "Yes").astype(int)
    df["DeviceProtection"] = (df["DeviceProtection"] == "Yes").astype(int)
    df["TechSupport"] = (df["TechSupport"] == "Yes").astype(int)
    df["StreamingTV"] = (df["StreamingTV"] == "Yes").astype(int)
    df["StreamingMovies"] = (df["StreamingMovies"] == "Yes").astype(int)

    df["M2M"] = (df["Contract"] == "Month-to-month").astype(int)
    df["OneYear"] = (df["Contract"] == "One year").astype(int)
    df.drop(columns=["Contract"], inplace=True)

    df["PaperlessBilling"] = (df["PaperlessBilling"] == "Yes").astype(int)

    df["Electronic Check"] = (df["PaymentMethod"] == "Electronic check").astype(int)
    df["Credit card (automatic)"] = (df["PaymentMethod"] == "Credit card (automatic)").astype(int)
    df["Bank transfer (automatic)"] = (df["PaymentMethod"] == "Bank transfer (automatic)").astype(int)
    df.drop(columns=["PaymentMethod"], inplace=True)

    df = df.reindex(columns=model_columns, fill_value=0)
    return df


@app.get("/")
def home():
    return {"message": "Churn API is running"}


@app.post("/predict")
def predict(customer: Customer):
    df = encode_customers([customer])
    prob = model.predict_proba(df)[0][1]
    return {"churn_probability": round(float(prob), 4)}


@app.post("/predict_batch")
def predict_batch(customers: List[Customer]):
    df = encode_customers(customers)
    probs = model.predict_proba(df)[:, 1]
    return [
        {"row": i, "churn_probability": round(float(p), 4)}
        for i, p in enumerate(probs)
    ]


