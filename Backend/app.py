# app.py
from fastapi import FastAPI
from sqlalchemy import create_engine, text
import pandas as pd
import pymysql

# Database connection details
user = "root"
password = "subhash"       # your MySQL password
host = "127.0.0.1"
port = "8080"              # replace with your actual MySQL port
database = "fraud_detection"

# Create SQLAlchemy engine
engine = create_engine(f"mysql+pymysql://{user}:{password}@{host}:{port}/{database}")

# Create FastAPI app
app = FastAPI()

# Root endpoint
@app.get("/")
def root():
    return {"message": "Welcome to Fraud Detection API"}

# Fetch sample transactions
@app.get("/api/transactions")
def get_transactions(limit: int = 10):
    query = text(f"SELECT * FROM transactions LIMIT {limit};")
    with engine.connect() as conn:
        df = pd.read_sql(query, conn)

    if df.empty:
        return {"message": "No transactions found in the database"}
    return df.to_dict(orient="records")

# Insert a new transaction
@app.post("/api/transactions")
def add_transaction(data: dict):
    new_transaction = pd.DataFrame([data])
    with engine.connect() as conn:
        new_transaction.to_sql("transactions", con=conn, if_exists="append", index=False)
    return {"message": "Transaction added successfully!"}
