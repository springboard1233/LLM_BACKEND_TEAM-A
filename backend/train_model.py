import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# Load training data
df = pd.read_csv("C:\\Users\\bharath\\Desktop\\ISB\\BFSI\\Backend\\data\\splits\\train.csv")

# Drop transaction_id (not useful for training)
if "transaction_id" in df.columns:
    df = df.drop(columns=["transaction_id"])

# Separate features and target
X = df.drop(columns=["is_fraud"])
y = df["is_fraud"]

# Encode categorical columns
label_encoders = {}
for col in X.select_dtypes(include=["object"]).columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le

# Split for training/testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model & encoders
joblib.dump({"model": model, "encoders": label_encoders}, "fraud_model.pkl")

print("Model trained and saved as fraud_model.pkl")
