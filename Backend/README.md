# 📘 Fraud Detection Project

This project focuses on detecting fraudulent financial transactions using data preprocessing, exploratory data analysis (EDA), and a FastAPI backend API.


## Project Structure

fraud_detection_project/
│── app.py                               # FastAPI API backend
│
├── notebooks/
│   └── LLM(1).ipynb                     # Preprocessing & EDA work
│
├── data/
│   ├── fraud_detection_dataset_LLM.csv  # Raw dataset
│   ├── processed_transactions.csv       # Processed dataset after cleaning
│   ├── train.csv                        # Training split
│   └── test.csv                         # Testing split

## Exploratory Data Analysis (LLM(1).ipynb)

The Jupyter notebook contains:

* **Data preprocessing**

  * Handling missing values
  * Removing duplicates
  * Feature engineering
* **EDA**

  * Summary statistics
  * Visualizations of transaction amounts
  * Class imbalance analysis (fraud vs. non-fraud)
* **Outputs**

  * processed_transactions.csv (cleaned data)
  * train.csv` & `test.csv (train-test split)


## Backend API (`app.py`)

A FastAPI backend that connects to a MySQL database and exposes transaction data.

### Endpoints

* **Root**


  GET /

  Returns:
  {"message": "Welcome to Fraud Detection API"}

* **Fetch Transactions**

  GET /api/transactions?limit=10

* **Insert Transaction**

  POST /api/transactions

  Example body:

  {
    "transaction_id": 12345,
    "amount": 250.75,
    "type": "payment",
    "is_fraud": 0
  }

## Setup & Run

1. Install dependencies:

   pip install fastapi uvicorn pandas sqlalchemy pymysql

2. Run the API:

   uvicorn app:app --reload

3. Access docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)



