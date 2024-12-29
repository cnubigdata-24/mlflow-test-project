import argparse
import os
import numpy as np
import pandas as pd
import mlflow
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder

# Set MLflow tracking URI and experiment
mlflow.set_tracking_uri("file:D:/mlflow-project/mlflow_test_project/mlruns")
mlflow.set_experiment("example_project")

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, required=True)
parser.add_argument("--max_iter", type=int, default=700)
parser.add_argument("--solver", type=str, default="lbfgs")
parser.add_argument("--penalty", type=str, default="l2")
parser.add_argument("--C", type=float, default=1.0)
args = parser.parse_args()

# Check if data file exists, if not generate and save data
#if not os.path.exists(args.data_path):
#np.random.seed(42)
data = {
    'age': np.random.randint(18, 95, size=1000),
    'income': np.random.randint(20000, 100000, size=1000),
    'education_level': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], size=1000),
    'employment_status': np.random.choice(['Employed', 'Unemployed', 'Self-employed'], size=1000),
    'marital_status': np.random.choice(['Single', 'Married', 'Divorced'], size=1000),
    'house_ownership': np.random.choice(['Own', 'Rent', 'Mortgage'], size=1000),
    'credit_score': np.random.randint(300, 850, size=1000),
    'target': np.random.randint(0, 2, size=1000)
}
df = pd.DataFrame(data)
df.to_csv(args.data_path, index=False)

# Load data
data = pd.read_csv(args.data_path)

# Encode categorical features
categorical_cols = ['education_level', 'employment_status', 'marital_status', 'house_ownership']
for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])

X = data.drop("target", axis=1)
y = data["target"]

# Train model
model = LogisticRegression(max_iter=args.max_iter, solver=args.solver, penalty=args.penalty, C=args.C)
with mlflow.start_run():

    # Log code file
    mlflow.log_artifact(__file__)

    model.fit(X, y)

    # Predict and evaluate
    predictions = model.predict(X)
    accuracy = accuracy_score(y, predictions)
    precision = precision_score(y, predictions)
    recall = recall_score(y, predictions)
    f1 = f1_score(y, predictions)
    roc_auc = roc_auc_score(y, model.predict_proba(X)[:, 1])

    # Log parameters, metrics, and model
    mlflow.log_param("max_iter", args.max_iter)
    mlflow.log_param("solver", args.solver)
    mlflow.log_param("penalty", args.penalty)
    mlflow.log_param("C", args.C)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("roc_auc", roc_auc)
    mlflow.sklearn.log_model(model, "model")
