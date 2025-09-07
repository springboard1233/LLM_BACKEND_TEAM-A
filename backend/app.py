from fastapi import FastAPI, Depends, HTTPException, Form, Query
from sqlalchemy import create_engine, Column, Integer, String, DECIMAL, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from passlib.context import CryptContext
from datetime import datetime
import pandas as pd
import pymysql
from pydantic import BaseModel

# ===============================
# Database Config
# ===============================
DB_USER = "root"
DB_PASS = "subhash"  # Change if needed
DB_HOST = "127.0.0.1"
DB_PORT = "8080"     
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

# Uncomment if you want to auto-create tables (optional)
# Base.metadata.create_all(bind=engine)

# ===============================
# Pydantic Models for Validation
# ===============================

class TransactionCreate(BaseModel):
    transaction_id: str
    customer_id: str
    kyc_verified: str
    account_age_days: int
    transaction_amount: float
    channel: str
    timestamp: str
    is_fraud: int
    hour: int
    day: int
    month: int
    weekday: int
    is_high_value: int

# ===============================
# FastAPI App
# ===============================
app = FastAPI()

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

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
def add_transaction(transaction: TransactionCreate):
    try:
        df = pd.DataFrame([transaction.dict()])
        # Use engine.begin() to ensure commit
        with engine.begin() as conn:
            df.to_sql("transactions", con=conn, if_exists="append", index=False)
        return {"message": "Transaction added successfully!"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error adding transaction: {e}")
