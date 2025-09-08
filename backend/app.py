from fastapi import FastAPI, Depends, HTTPException, Form, Query
from sqlalchemy import create_engine, Column, Integer, String, DECIMAL, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from passlib.context import CryptContext
from datetime import datetime
import pandas as pd
import pymysql
from pydantic import BaseModel
import joblib
import numpy as np

# ===============================
# Database Config
# ===============================
DB_USER = "root"
DB_PASS = "1234"  # Change if needed
DB_HOST = "127.0.0.1"
DB_PORT = "3306"
DB_NAME = "fraud_detection"

DATABASE_URL = f"mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# ===============================
# Password Hashing
# ===============================
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# ===============================
# Models
# ===============================
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    firstName = Column(String(100))
    lastName = Column(String(100))
    email = Column(String(255), unique=True, index=True)
    mobile = Column(String(20))
    countryCode = Column(String(10))
    password = Column(String(255))
    pin = Column(String(10))

class Transaction(Base):
    __tablename__ = "transactions"
    transaction_id = Column(String, primary_key=True)
    customer_id = Column(String)
    kyc_verified = Column(String)
    account_age_days = Column(Integer)
    transaction_amount = Column(DECIMAL(18, 4))
    channel = Column(String)
    timestamp = Column(String)
    is_fraud = Column(Integer)
    hour = Column(Integer)
    day = Column(Integer)
    month = Column(Integer)
    weekday = Column(Integer)
    is_high_value = Column(Integer)

# Uncomment for auto-create tables
# Base.metadata.create_all(bind=engine)

# ===============================
# Pydantic Models
# ===============================
class TransactionPredict(BaseModel):
    customer_id: str
    kyc_verified: str
    account_age_days: int
    transaction_amount: float
    channel: str
    timestamp: str
    hour: int
    day: int
    month: int
    weekday: int
    is_high_value: int

# ===============================
# FastAPI App
# ===============================
app = FastAPI()

# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ===============================
# Load Model
# ===============================
try:
    model_bundle = joblib.load("fraud_model.pkl")
    model = model_bundle["model"]
    encoders = model_bundle["encoders"]
except Exception as e:
    model, encoders = None, {}
    print(f"⚠️ Warning: Could not load model - {e}")

# ===============================
# Auth Routes
# ===============================
@app.post("/signup")
def signup(
    email: str = Form(...),
    password: str = Form(...),
    firstName: str = Form(None),
    lastName: str = Form(None),
    mobile: str = Form(None),
    countryCode: str = Form(None),
    pin: str = Form(None),
    db: Session = Depends(get_db)
):
    if db.query(User).filter(User.email == email).first():
        raise HTTPException(status_code=400, detail="Email already exists")

    hashed_pw = pwd_context.hash(password)
    new_user = User(
        email=email,
        password=hashed_pw,
        firstName=firstName,
        lastName=lastName,
        mobile=mobile,
        countryCode=countryCode,
        pin=pin
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return {"message": "User created successfully", "user_id": new_user.id}

@app.post("/login")
def login(email: str = Form(...), password: str = Form(...), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == email).first()
    if not user or not pwd_context.verify(password, user.password):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    return {"message": "Login successful", "user_id": user.id}

# ===============================
# Transaction Routes
# ===============================
@app.get("/")
def root():
    return {"message": "Welcome to Fraud Detection API"}

@app.get("/api/transactions")
def get_transactions(limit: int = Query(10, ge=1)):
    query = text(f"SELECT * FROM transactions LIMIT {limit}")
    with engine.connect() as conn:
        df = pd.read_sql(query, conn)
    if df.empty:
        return {"message": "No transactions found in the database"}
    return df.to_dict(orient="records")

@app.post("/api/transactions")
def add_transaction(transaction: TransactionPredict):
    if not model:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        data = transaction.dict()
        df = pd.DataFrame([data])

        # Apply label encoders
        for col, le in encoders.items():
            if col in df.columns:
                df[col] = le.transform(df[col].astype(str))

        # Run prediction
        prediction = model.predict(df)[0]
        probability = model.predict_proba(df)[0][1]

        # Add prediction result
        data["is_fraud"] = int(prediction)
        df_final = pd.DataFrame([data])

        with engine.begin() as conn:
            df_final.to_sql("transactions", con=conn, if_exists="append", index=False)

        return {
            "message": "Transaction added successfully!",
            "prediction": int(prediction),
            "fraud_probability": float(probability)
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error adding transaction: {e}")

# ===============================
# Prediction API
# ===============================
@app.post("/api/predict")
def predict_fraud(transaction: TransactionPredict):
    if not model:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        data = transaction.dict()
        df = pd.DataFrame([data])

        # Apply label encoders
        for col, le in encoders.items():
            if col in df.columns:
                df[col] = le.transform(df[col].astype(str))

        prediction = model.predict(df)[0]
        probability = model.predict_proba(df)[0][1]

        return {
            "prediction": int(prediction),
            "fraud_probability": float(probability)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {e}")

# ===============================
# Fraud Alerts API
# ===============================
@app.get("/api/fraud-alerts")
def get_fraud_alerts(limit: int = Query(10, ge=1)):
    query = text(f"SELECT * FROM transactions WHERE is_fraud = 1 ORDER BY timestamp DESC LIMIT {limit}")
    with engine.connect() as conn:
        df = pd.read_sql(query, conn)
    if df.empty:
        return {"message": "No fraud alerts found"}
    return df.to_dict(orient="records")

# ===============================
# Fraud Stats API
# ===============================
@app.get("/api/fraud-stats")
def fraud_stats():
    query = text("SELECT is_fraud, COUNT(*) as count FROM transactions GROUP BY is_fraud")
    with engine.connect() as conn:
        df = pd.read_sql(query, conn)

    if df.empty:
        return {"message": "No transactions available"}

    total = int(df["count"].sum())
    fraud = int(df.loc[df["is_fraud"] == 1, "count"].sum()) if 1 in df["is_fraud"].values else 0
    legit = int(df.loc[df["is_fraud"] == 0, "count"].sum()) if 0 in df["is_fraud"].values else 0

    return {
        "total_transactions": total,
        "fraud_transactions": fraud,
        "legit_transactions": legit,
        "fraud_percentage": round((fraud / total) * 100, 2) if total > 0 else 0
    }

