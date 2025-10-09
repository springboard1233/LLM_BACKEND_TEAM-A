from fastapi import FastAPI, Depends, HTTPException, Form, Query
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, Column, Integer, String, DECIMAL, Float, DateTime, text, func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from passlib.context import CryptContext
from datetime import datetime
import pandas as pd
import pymysql
from pydantic import BaseModel
import joblib
import numpy as np
import uuid
from pathlib import Path
from decimal import Decimal

# --- NEW LLM IMPORTS and CONFIG ---
import os
import asyncio
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Load environment variables (reads GEMINI_API_KEY from .env file)
load_dotenv() 

# ===============================
# Database Config (FIXED DB PORT)
# ===============================
DB_USER = "root"
DB_PASS = "Varun23141"  # Change if needed
DB_HOST = "127.0.0.1"
DB_PORT = "3306"  # CORRECTED to default MySQL port
DB_NAME = "fraud_detection"

DATABASE_URL = f"mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
try:
    engine = create_engine(DATABASE_URL)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base = declarative_base()
except Exception as e:
    print(f"ERROR: Could not create database engine. Check your MySQL server and credentials. Error: {e}")
    Base = declarative_base()

# ===============================
# LLM Client Initialization
# ===============================
llm_client = None
try:
    # Initialize the Gemini client. It finds the GEMINI_API_KEY automatically.
    llm_client = genai.Client() 
except Exception as e:
    print(f"Warning: Could not initialize LLM client. Ensure 'google-genai' package is installed and API key is set. Error: {e}")


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
    transaction_id = Column(String(50), primary_key=True)
    customer_id = Column(String(50))
    kyc_verified = Column(String(10))
    account_age_days = Column(Integer)
    transaction_amount = Column(DECIMAL(18, 4))
    channel = Column(String(50))
    timestamp = Column(String(50)) 
    is_fraud = Column(Integer)
    hour = Column(Integer)
    day = Column(Integer)
    month = Column(Integer)
    weekday = Column(Integer)
    is_high_value = Column(Integer)

class FraudAlert(Base):
    __tablename__ = "fraud_alerts"
    alert_id = Column(Integer, primary_key=True, autoincrement=True)
    transaction_id = Column(String(50))
    customer_id = Column(String(50))
    risk_score = Column(Float)
    reason = Column(String(512)) # Increased length
    created_at = Column(DateTime, server_default=text('CURRENT_TIMESTAMP'))

# Auto-create tables if they don't exist
try:
    Base.metadata.create_all(bind=engine)
except NameError:
    print("WARNING: Skipped table creation due to database connection issue.")

# ===============================
# Pydantic Models
# ===============================
class TransactionPredict(BaseModel):
    transaction_id: str
    customer_id: str
    kyc_verified: str
    account_age_days: int
    transaction_amount: float
    channel: str
    timestamp: str
    is_fraud: str
    hour: int
    day: int
    month: int
    weekday: int
    is_high_value: int

# ===============================
# FastAPI App
# ===============================
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ===============================
# Load Model & Metrics
# ===============================
try:
    BASE_DIR = Path(__file__).resolve().parent
    MODEL_PATH = BASE_DIR / "fraud_model_rf.pkl"
    model_bundle = joblib.load(MODEL_PATH)
    model = model_bundle["model"]
    encoders = model_bundle["encoders"]
    metrics = model_bundle.get("metrics", {})
except Exception as e:
    model, encoders, metrics = None, {}, {}
    print(f"⚠ Warning: Could not load model - {e}")


# ===============================
# LLM/Explanation Helper Function (LIVE GEMINI API Call)
# ===============================

def _generate_llm_explanation_gemini(alert_reason: str, risk_score: float, customer_id: str) -> str:
    """
    Synchronous function to call the Gemini API and generate the explanation.
    This function will be run in a separate thread using asyncio.to_thread.
    """
    if not llm_client:
        return f"LLM client is not available. Raw Reason: {alert_reason}"

    # Craft a clear prompt using the structured data
    prompt = f"""
    You are a professional financial fraud detection analyst. A transaction for customer ID {customer_id}
    was flagged with a combined risk score of {risk_score:.2f}.

    The raw technical reasons for the alert are: "{alert_reason}".

    Generate a clear, one-paragraph summary for an internal reviewer, explaining the primary
    reasons for the fraud alert and why the risk score is high. Be concise and professional.
    """
    
    try:
        # Use the generate_content method
        response = llm_client.models.generate_content(
            model='gemini-2.5-flash',  # A fast and capable model for this task
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.2  # Keep creativity low for technical explanations
            )
        )
        return response.text
    except Exception as e:
        print(f"Error calling Gemini API: {e}")
        return f"Error: Could not retrieve LLM explanation from Gemini. Raw Reason: {alert_reason}"

# ===============================
# Auth Routes (Unchanged)
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
# Transaction Routes & Prediction Helpers (Unchanged logic)
# ===============================
@app.get("/")
def root():
    return {"message": "Welcome to Fraud Detection API"}

@app.get("/api/transactions")
def get_transactions(limit: int = Query(10, ge=1)):
    query = text(f"SELECT * FROM transactions ORDER BY timestamp DESC LIMIT {limit}")
    with engine.connect() as conn:
        df = pd.read_sql(query, conn)
    if df.empty:
        return {"message": "No transactions found in the database"}
    df['transaction_amount'] = df['transaction_amount'].astype(float)
    return df.to_dict(orient="records")

@app.post("/api/transactions")
def add_transaction(transaction: TransactionPredict, db: Session = Depends(get_db)):
    if not model:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        data = transaction.dict()
        df = pd.DataFrame([data])

        df_predict = df.drop(columns=['transaction_id', 'is_fraud'], errors='ignore')

        for col, le in encoders.items():
            if col in df_predict.columns:
                try:
                    df_predict[col] = le.transform(df_predict[col].astype(str))
                except ValueError:
                    print(f"Warning: Unseen label in column '{col}'.")
                    df_predict[col] = -1

        prediction = model.predict(df_predict)[0]
        probability = model.predict_proba(df_predict)[0][1]

        # Prepare ORM object for reliable persistence
        new_tx = Transaction(
            transaction_id=str(uuid.uuid4()),
            customer_id=data.get("customer_id"),
            kyc_verified=str(data.get("kyc_verified")),
            account_age_days=int(data.get("account_age_days")),
            transaction_amount=Decimal(str(data.get("transaction_amount"))),
            channel=str(data.get("channel")),
            timestamp=str(data.get("timestamp")),
            is_fraud=int(prediction),
            hour=int(data.get("hour")),
            day=int(data.get("day")),
            month=int(data.get("month")),
            weekday=int(data.get("weekday")),
            is_high_value=int(data.get("is_high_value")),
        )

        db.add(new_tx)
        db.commit()
        db.refresh(new_tx)

        return {
            "message": "Transaction added successfully!",
            "prediction": int(prediction),
            "fraud_probability": float(probability),
            "transaction_id": new_tx.transaction_id
        }

    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=400, detail=f"Error adding transaction: {e}")

def _get_avg_amount_for_customer(db: Session, customer_id: str) -> float:
    # Try customer-specific average
    cust_avg = db.query(func.avg(Transaction.transaction_amount)).filter(Transaction.customer_id == customer_id).scalar()
    if cust_avg is not None:
        try:
            return float(cust_avg)
        except Exception:
            pass
    # Fallback to global average
    global_avg = db.query(func.avg(Transaction.transaction_amount)).scalar()
    if global_avg is not None:
        try:
            return float(global_avg)
        except Exception:
            pass
    # Final fallback
    return 0.0

def _evaluate_rules(payload: dict, avg_amount: float) -> list:
    rules_triggered = []
    try:
        amount = float(payload.get("transaction_amount", 0) or 0)
    except Exception:
        amount = 0.0
    channel = str(payload.get("channel", "")).lower()
    kyc_verified = str(payload.get("kyc_verified", "")).lower()
    hour = int(payload.get("hour", 0) or 0)

    # Rule 1: High amount vs average
    if avg_amount > 0 and amount > 5 * avg_amount:
        rules_triggered.append("Amount > 5x average")

    # Rule 2: International + KYC not verified
    if channel == "international" and kyc_verified in {"no", "false", "0"}:
        rules_triggered.append("International + KYC not verified")

    # Rule 3: Odd hours 2AM-4AM
    if 2 <= hour <= 4:
        rules_triggered.append("Odd transaction hour (2AM-4AM)")

    return rules_triggered

def _combine_risk(model_prob: float, rules_triggered: list) -> float:
    # Simple combination: add 0.2 per triggered rule, cap at 1.0
    bump = 0.2 * len(rules_triggered)
    risk = min(1.0, model_prob + bump)
    return float(risk)

def _make_decision(model_prob: float, rules_triggered: list, threshold: float) -> bool:
    # Decide fraud if model prob crosses threshold OR any rule triggers
    return (model_prob >= threshold) or (len(rules_triggered) > 0)

@app.post("/predict")
def predict_realtime(transaction: TransactionPredict, db: Session = Depends(get_db)):
    if not model:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        data = transaction.dict()
        df = pd.DataFrame([data])

        # Preprocess like training
        df_predict = df.drop(columns=["transaction_id", "is_fraud"], errors="ignore")
        for col, le in encoders.items():
            if col in df_predict.columns:
                try:
                    df_predict[col] = le.transform(df_predict[col].astype(str))
                except ValueError:
                    # unseen label -> -1
                    df_predict[col] = -1

        model_pred = model.predict(df_predict)[0]
        model_prob = float(model.predict_proba(df_predict)[0][1])

        # Rule engine
        avg_amount = _get_avg_amount_for_customer(db, data.get("customer_id"))
        rules_triggered = _evaluate_rules(data, avg_amount)

        # Combine
        threshold = float(metrics.get("threshold", 0.5)) if metrics else 0.5
        risk_score = _combine_risk(model_prob, rules_triggered)
        is_fraud_final = _make_decision(model_prob, rules_triggered, threshold)

        result_label = "Fraud" if is_fraud_final else "Legit"
        message_parts = []
        if is_fraud_final:
            if model_prob >= threshold:
                message_parts.append("Model anomaly")
            if rules_triggered:
                message_parts.append("; ".join(rules_triggered))
        message = " | ".join(message_parts) if message_parts else "No anomalies detected"

        # Persist transaction prediction result
        incoming_tx_id = str(data.get("transaction_id") or "").strip()
        tx_id = incoming_tx_id if incoming_tx_id and incoming_tx_id.lower() not in {"temp", "placeholder", "na"} else str(uuid.uuid4())
        new_tx = Transaction(
            transaction_id=tx_id,
            customer_id=str(data.get("customer_id")),
            kyc_verified=str(data.get("kyc_verified")),
            account_age_days=int(data.get("account_age_days") or 0),
            transaction_amount=Decimal(str(data.get("transaction_amount") or 0)),
            channel=str(data.get("channel")),
            timestamp=str(data.get("timestamp")),
            is_fraud=1 if is_fraud_final else 0,
            hour=int(data.get("hour") or 0),
            day=int(data.get("day") or 0),
            month=int(data.get("month") or 0),
            weekday=int(data.get("weekday") or 0),
            is_high_value=int(data.get("is_high_value") or 0),
        )
        db.merge(new_tx) 
        db.commit()

        # Persist alert if flagged
        if is_fraud_final:
            try:
                alert = FraudAlert(
                    transaction_id=tx_id,
                    customer_id=str(data.get("customer_id")),
                    risk_score=round(risk_score, 2),
                    reason=message or "rule triggered"
                )
                db.add(alert)
                db.commit()
            except Exception as e:
                db.rollback()
                print(f"Warning: Failed to write fraud alert: {e}")

        return {
            "transaction_id": tx_id,
            "prediction": result_label,
            "risk_score": round(risk_score, 2),
            "reason": message if message else ("High risk" if is_fraud_final else "Low risk")
        }
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=400, detail=f"Prediction error: {e}")

# ===============================
# LLM Explanation API (LIVE GEMINI)
# ===============================
@app.get("/api/llm-explain/{transaction_id}")
async def get_llm_explanation(transaction_id: str, db: Session = Depends(get_db)):
    """
    API for LLM explanation/insights (Now using live Gemini API call).
    The async keyword ensures the thread is not blocked while waiting for the Gemini API response.
    """
    alert = db.query(FraudAlert).filter(FraudAlert.transaction_id == transaction_id).first()
    
    if not alert:
        return {
            "transaction_id": transaction_id,
            "status": "No Alert",
            "explanation": "No fraud alert was recorded for this transaction. It was processed as legitimate."
        }

    # IMPORTANT: Use asyncio.to_thread to run the synchronous Gemini call 
    # in a separate thread, preventing the FastAPI server from freezing.
    llm_explanation = await asyncio.to_thread(
        _generate_llm_explanation_gemini,
        alert.reason, 
        alert.risk_score, 
        alert.customer_id
    )

    
    return {
        "transaction_id": transaction_id,
        "risk_score": alert.risk_score,
        "raw_reason": alert.reason,
        "explanation": llm_explanation
    }

# ===============================
# Fraud Alerts API (Unchanged)
# ===============================
@app.get("/api/fraud-alerts")
def get_fraud_alerts(limit: int = Query(10, ge=1)):
    query = text("SELECT * FROM fraud_alerts ORDER BY created_at DESC LIMIT :limit")
    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params={"limit": limit})
    if df.empty:
        return {"message": "No fraud alerts found"}
    return df.to_dict(orient="records")

# ===============================
# Fraud Stats API (Unchanged)
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

# ===============================
# Dashboard Analytics API (Unchanged)
# ===============================
@app.get("/api/analytics/channels")
def get_channel_fraud_distribution():
    query = text("SELECT channel, COUNT(*) as total_count, SUM(is_fraud) as fraud_count FROM transactions GROUP BY channel")
    with engine.connect() as conn:
        df = pd.read_sql(query, conn)
    
    if df.empty:
        return {"message": "No channel data available"}

    df['fraud_percentage'] = (df['fraud_count'] / df['total_count']) * 100
    
    return df.to_dict(orient="records")

@app.get("/api/analytics/timeseries")
def get_time_series_fraud():
    query = text("SELECT DATE(timestamp) as transaction_date, COUNT(*) as total_count, SUM(is_fraud) as fraud_count FROM transactions GROUP BY transaction_date ORDER BY transaction_date")
    with engine.connect() as conn:
        df = pd.read_sql(query, conn)
    
    if df.empty:
        return {"message": "No time series data available"}
    
    df['fraud_percentage'] = (df['fraud_count'] / df['total_count']) * 100
    df['transaction_date'] = df['transaction_date'].astype(str)
    
    return df.to_dict(orient="records")

@app.get("/api/analytics/amount_vs_fraud")
def get_amount_vs_fraud():
    query = text("SELECT transaction_amount, is_fraud FROM transactions")
    with engine.connect() as conn:
        df = pd.read_sql(query, conn)

    if df.empty:
        return {"message": "No data available"}
    
    df['transaction_amount'] = df['transaction_amount'].astype(float)
    return df.to_dict(orient="records")

# ===============================
# Model Metrics API (Unchanged)
# ===============================
@app.get("/api/model-info")
def get_model_info():
    if not metrics:
        raise HTTPException(status_code=500, detail="Model metrics not loaded")
    safe_metrics = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in metrics.items()}
    return safe_metrics

# ===============================
# Maintenance / Admin APIs (Unchanged)
# ===============================
@app.delete("/api/transactions")
def delete_all_transactions(db: Session = Depends(get_db)):
    try:
        db.query(FraudAlert).delete()
        db.query(Transaction).delete()
        db.commit()
        return {"message": "All transactions and fraud alerts deleted"}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to delete transactions: {e}")
