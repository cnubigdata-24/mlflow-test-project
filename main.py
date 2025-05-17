#pip install mlflow scikit-learn pandas numpy pyyaml

import os
import numpy as np
import pandas as pd
import mlflow
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder
import yaml
import argparse
from mlflow.models.signature import infer_signature

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, default="data/input.csv", help="Path to the data file")
args = parser.parse_args()

# Use command line arguments instead of reading from MLproject
data_path = args.data_path
max_iter = 700
solver = "lbfgs"
penalty = "l2"
C = 1.0

# Create data directory
data_dir = os.path.dirname(data_path)
if data_dir and not os.path.exists(data_dir):
    os.makedirs(data_dir)

# Check if data file exists, if not generate and save data
if not os.path.exists(data_path):
    np.random.seed(42)
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
    df.to_csv(data_path, index=False)

data = pd.read_csv(data_path)

# Encode categorical features
categorical_cols = ['education_level', 'employment_status', 'marital_status', 'house_ownership']
for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])

# Convert integer columns to float
numerical_cols = ['age', 'income', 'credit_score']
for col in numerical_cols:
    data[col] = data[col].astype(float)

X = data.drop("target", axis=1)
y = data["target"]

# Train model
model = LogisticRegression(max_iter=max_iter, solver=solver, penalty=penalty, C=C)

# No need to use mlflow.start_run() when running as MLflow project
# MLflow automatically creates a run
model.fit(X, y)

# Predict and evaluate
predictions = model.predict(X)
accuracy = accuracy_score(y, predictions)
precision = precision_score(y, predictions)
recall = recall_score(y, predictions)
f1 = f1_score(y, predictions)
roc_auc = roc_auc_score(y, model.predict_proba(X)[:, 1])

# Log parameters, metrics, and model
mlflow.log_param("data_path", data_path)
mlflow.log_param("max_iter", max_iter)
mlflow.log_param("solver", solver)
mlflow.log_param("penalty", penalty)
mlflow.log_param("C", C)
mlflow.log_metric("accuracy", accuracy)
mlflow.log_metric("precision", precision)
mlflow.log_metric("recall", recall)
mlflow.log_metric("f1_score", f1)
mlflow.log_metric("roc_auc", roc_auc)

# Create model signature and input example
signature = infer_signature(X, predictions)
input_example = X.iloc[:5]  

# Log model with signature and input example
mlflow.sklearn.log_model(
    model, 
    "model",
    signature=signature,
    input_example=input_example
)

# Log model summary information
model_info = {
    "model_type": "LogisticRegression",
    "feature_count": X.shape[1],
    "training_samples": X.shape[0],
    "categorical_features": categorical_cols,
    "numerical_features": numerical_cols
}

# Save model info to temporary file and log it
model_info_path = "model_info.yaml"
with open(model_info_path, 'w') as f:
    yaml.dump(model_info, f)

mlflow.log_artifact(model_info_path)
os.remove(model_info_path)  # Clean up temporary file

# Log MLproject and conda.yaml files
if os.path.exists("MLproject"):
    mlflow.log_artifact("MLproject")

if os.path.exists("conda.yaml"):
    mlflow.log_artifact("conda.yaml")

# Log generated data file
mlflow.log_artifact(data_path)

# Log feature importance
if hasattr(model, 'coef_'):
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': np.abs(model.coef_[0])
    }).sort_values('importance', ascending=False)
    
    feature_importance_path = "feature_importance.csv"
    feature_importance.to_csv(feature_importance_path, index=False)
    mlflow.log_artifact(feature_importance_path)
    os.remove(feature_importance_path)  # Clean up temporary file

print(f"MLflow run completed successfully! Data path: {data_path}")

# mlflow run . --env-manager=local --experiment-name "new_example_project"
