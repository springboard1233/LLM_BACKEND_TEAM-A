import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
import joblib
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# ===============================
# Load dataset (UPDATED PATH LOGIC)
# ===============================
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH_TRAIN = BASE_DIR / "train.csv"
DATA_PATH_FRAUD = BASE_DIR / "fraud_dataset_10000.csv"

if DATA_PATH_TRAIN.exists():
    DATA_PATH = DATA_PATH_TRAIN
elif DATA_PATH_FRAUD.exists():
    DATA_PATH = DATA_PATH_FRAUD
else:
    print("FATAL ERROR: Could not find train.csv or fraud_dataset_10000.csv.")
    raise FileNotFoundError("Training data not found. Please ensure one of the CSV files is in the same directory.")

df = pd.read_csv(DATA_PATH)

# Drop transaction_id if present
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

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ===============================
# Handle imbalance with SMOTE
# ===============================
print("Applying SMOTE to balance classes...")
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# ===============================
# Random Forest Classifier
# ===============================
rf = RandomForestClassifier(
    random_state=42,
    class_weight="balanced_subsample",  # penalize majority class
    n_jobs=-1
)

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, 15],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=3,
    scoring='f1',   # optimize for fraud detection
    n_jobs=-1,
    verbose=1
)

print("Starting hyperparameter tuning...")
grid_search.fit(X_train, y_train)

model = grid_search.best_estimator_
print("\nBest Model Parameters:", grid_search.best_params_)

# ===============================
# Evaluation with threshold tuning
# ===============================
y_proba = model.predict_proba(X_test)[:, 1]

# Adjust threshold from 0.5 → 0.3 (catch more fraud)
threshold = 0.3
y_pred = (y_proba >= threshold).astype(int)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, output_dict=True)
cm = confusion_matrix(y_test, y_pred)
auc_score = roc_auc_score(y_test, y_proba)
fpr, tpr, thresholds = roc_curve(y_test, y_proba)

# Metrics
model_metrics = {
    "accuracy": accuracy,
    "precision": report["1"]["precision"],
    "recall": report["1"]["recall"],
    "f1_score": report["1"]["f1-score"],
    "auc_score": auc_score,
    # Specificity = TN / (TN + FP)
    "specificity": (lambda cm_: (
        (cm_[0, 0] / (cm_[0, 0] + cm_[0, 1])) if (cm_[0, 0] + cm_[0, 1]) > 0 else 0.0
    ))(cm),
    "confusion_matrix": cm.tolist(),
    # Only store a subset of ROC points to keep model file size small, if needed for front end
    "roc_curve_subset": {"fpr": fpr[::50].tolist(), "tpr": tpr[::50].tolist()}, 
    "threshold": threshold
}

# Save model and encoders
joblib.dump(
    {"model": model, "encoders": label_encoders, "metrics": model_metrics},
    "fraud_model_rf1.pkl"
)

print(f"\nOverall Model Accuracy: {accuracy:.4f}")
print("\nClassification Report (threshold=0.3):")
print(classification_report(y_test, y_pred))

# ===============================
# Print Metrics Table
# ===============================
metrics_table = pd.DataFrame([
    {
        "Metric": "Accuracy",
        "Value": accuracy,
        "Description": "Overall correctness of predictions"
    },
    {
        "Metric": "Precision",
        "Value": report["1"]["precision"],
        "Description": "TP / (TP + FP)"
    },
    {
        "Metric": "Recall",
        "Value": report["1"]["recall"],
        "Description": "TP / (TP + FN)"
    },
    {
        "Metric": "F1-Score",
        "Value": report["1"]["f1-score"],
        "Description": "Harmonic mean of Precision & Recall"
    },
    {
        "Metric": "Specificity",
        "Value": model_metrics["specificity"],
        "Description": "TN / (TN + FP)"
    },
    {
        "Metric": "AUC Score",
        "Value": auc_score,
        "Description": "Area under ROC curve"
    },
])

print("\nPerformance Metrics Table:")
print(metrics_table.to_string(index=False, formatters={"Value": lambda v: f"{v:.4f}"}))

# ===============================
# Visualizations
# ===============================

# 1) ROC Curve
plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, label=f'AUC = {auc_score:.4f}', color='tab:blue')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Random Forest Model')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 2) Confusion Matrix Heatmap
tn, fp, fn, tp = cm.ravel()
cm_display = np.array([[tn, fp], [fn, tp]])
plt.figure(figsize=(5, 4))
plt.imshow(cm_display, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(2)
classes = ['Legit (0)', 'Fraud (1)']
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)

thresh = cm_display.max() / 2.
for i in range(2):
    for j in range(2):
        plt.text(j, i, format(cm_display[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm_display[i, j] > thresh else "black")

plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.tight_layout()
plt.show()

# 3) Fraud vs Legit (Bar Chart)
counts = df['is_fraud'].value_counts().sort_index()
labels = ['Legit (0)', 'Fraud (1)']
plt.figure(figsize=(6, 4))
plt.bar(labels, [counts.get(0, 0), counts.get(1, 0)], color=['tab:green', 'tab:red'])
plt.title('Fraud vs Legit - Count of Transactions')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

# 4) Fraud% over time (Line Chart)
if 'timestamp' in df.columns:
    df_time = df.copy()
    # Parse datetime and take date only
    df_time['date'] = pd.to_datetime(df_time['timestamp'], errors='coerce').dt.date
    ts = df_time.dropna(subset=['date']).groupby('date').agg(
        total=('is_fraud', 'count'),
        fraud=('is_fraud', 'sum')
    )
    if not ts.empty:
        fraud_pct = (ts['fraud'] / ts['total']) * 100
        plt.figure(figsize=(8, 4))
        plt.plot(fraud_pct.index, fraud_pct.values, marker='o', color='tab:orange')
        plt.title('Fraud Percentage Over Time')
        plt.xlabel('Date')
        plt.ylabel('Fraud %')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

# 5) Channel-Wise Fraud Distribution (Pie Chart)
if 'channel' in df.columns:
    ch = df.groupby('channel')['is_fraud'].sum().sort_values(ascending=False)
    if not ch.empty:
        plt.figure(figsize=(6, 6))
        plt.pie(ch.values, labels=ch.index, autopct='%1.1f%%', startangle=140)
        plt.title('Channel-Wise Fraud Distribution')
        plt.tight_layout()
        plt.show()

# 6) Transaction Amount vs Fraud (Box Plot)
if 'transaction_amount' in df.columns:
    plt.figure(figsize=(7, 5))
    # Prepare data
    amt_legit = df.loc[df['is_fraud'] == 0, 'transaction_amount']
    amt_fraud = df.loc[df['is_fraud'] == 1, 'transaction_amount']
    # showfliers=False to exclude extreme outliers for better visualization scale
    plt.boxplot([amt_legit, amt_fraud], labels=['Legit (0)', 'Fraud (1)'], showfliers=False) 
    plt.title('Transaction Amount vs Fraud (Without extreme outliers)')
    plt.ylabel('Transaction Amount')
    plt.tight_layout()
    plt.show()

print("\nRandom Forest model trained and saved as fraud_model_rf1.pkl with updated path logic.")