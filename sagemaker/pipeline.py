# monitor_pipeline.py
# Validates endpoint, runs baseline inference, computes baseline stats,
# induces drift, uploads baseline + drift to S3, and generates Evidently reports.

import re
import time
import json
import boto3
import pandas as pd
import numpy as np
from io import StringIO
from sklearn.metrics import mean_squared_error
from datetime import datetime

from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, RegressionPreset

import sagemaker
from sagemaker.model_monitor import DefaultModelMonitor, EndpointInput, DatasetFormat

# ------------------------------------
# Configuration
# ------------------------------------
REGION = "us-east-2"
ENDPOINT_NAME = "NYC-Taxi-Fare-Prod-Group-2025-12-11-16-25-35-688"

S3_BUCKET = "mlops-project-uoc"
S3_TEST_KEY = "processed/test.csv"
S3_TRAIN_KEY = "processed/train.csv"

S3_MONITOR_BASELINE_PREFIX = "monitoring/baseline"
S3_MONITOR_OUTPUT_PREFIX = "monitoring/output"
S3_REPORT_PREFIX = "monitoring/reports"

EXECUTION_ROLE_ARN = (
    "arn:aws:iam::428056471672:role/service-role/AmazonSageMakerAdminIAMExecutionRole"
)

target_col = "fare_amount"

required_features = [
    "key",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
    "hour",
    "month",
    "weekday",
    "distance_km"
]

BATCH_SIZE = 200

# ------------------------------------
# AWS client setup
# ------------------------------------
session = boto3.Session(region_name=REGION)
s3 = session.client("s3")
runtime = session.client("sagemaker-runtime")
sm = session.client("sagemaker")
sagemaker_session = sagemaker.Session(boto3.Session(region_name=REGION))

print(f"AWS clients initialized for region {REGION}")

# ------------------------------------
# Helpers
# ------------------------------------
def s3_uri(bucket, key):
    return f"s3://{bucket}/{key}"

def load_csv_from_s3(bucket, key):
    print(f"Loading s3://{bucket}/{key} ...")
    obj = s3.get_object(Bucket=bucket, Key=key)
    text = obj["Body"].read().decode("utf-8")
    return pd.read_csv(StringIO(text))

def df_to_csv_rows(df, feature_cols):
    rows = []
    for _, row in df[feature_cols].iterrows():
        vals = [
            ("" if pd.isna(v) else str(v).replace("\n", " ").replace("\r", " "))
            for v in row.values
        ]
        rows.append(",".join(vals))
    return rows

def parse_predictions_from_response(resp_bytes):
    raw = resp_bytes.decode("utf-8").strip()
    if raw == "":
        return []

    # Try JSON
    try:
        j = json.loads(raw)
        if isinstance(j, dict) and "predictions" in j:
            out = []
            for p in j["predictions"]:
                if isinstance(p, list):
                    out.extend([float(x) for x in p])
                else:
                    out.append(float(p))
            return out
        if isinstance(j, list):
            return [float(x) for x in j]
    except Exception:
        pass

    # Fallback CSV / numeric parsing
    out = []
    lines = raw.splitlines()
    for ln in lines:
        nums = re.findall(r"[-+]?\d*\.\d+|\d+", ln)
        out.extend([float(n) for n in nums])
    return out

def predict_batch_runtime(df_features, batch_size=BATCH_SIZE):
    rows = df_to_csv_rows(df_features, df_features.columns.tolist())
    preds = []

    for i in range(0, len(rows), batch_size):
        chunk = rows[i:i+batch_size]
        payload = "\n".join(chunk).encode("utf-8")

        resp = runtime.invoke_endpoint(
            EndpointName=ENDPOINT_NAME,
            ContentType="text/csv",
            Body=payload,
        )

        preds.extend(parse_predictions_from_response(resp["Body"].read()))

    return preds

# ------------------------------------
# Load test data
# ------------------------------------
df_test = load_csv_from_s3(S3_BUCKET, S3_TEST_KEY)

# Validate columns
missing = [c for c in required_features + [target_col] if c not in df_test.columns]
if missing:
    raise RuntimeError("Missing columns: " + ", ".join(missing))

X_test = df_test[required_features].copy()

# Convert numeric fields
numeric_cols = [c for c in required_features if c not in ["key", "pickup_datetime"]]

for c in numeric_cols:
    X_test[c] = pd.to_numeric(X_test[c], errors="coerce")

X_test["pickup_datetime"] = X_test["pickup_datetime"].astype(str)
X_test["key"] = X_test["key"].astype(str)
X_test = X_test.fillna(0)

y_test = df_test[target_col].copy()

print("Prepared X_test:", X_test.shape)

# ------------------------------------
# Baseline inference
# ------------------------------------
print("Running baseline inference...")
baseline_preds = predict_batch_runtime(X_test)

df_baseline = df_test.copy()
df_baseline["prediction"] = baseline_preds
df_baseline["target"] = df_baseline[target_col]

rmse_baseline = mean_squared_error(df_baseline["target"], df_baseline["prediction"]) ** 0.5
print(f"BASELINE RMSE: {rmse_baseline:.4f}")

# ------------------------------------
# Upload baseline
# ------------------------------------
baseline_local = "/tmp/baseline.csv"
df_baseline.to_csv(baseline_local, index=False)

baseline_key = f"{S3_MONITOR_BASELINE_PREFIX}/baseline.csv"
s3.upload_file(baseline_local, S3_BUCKET, baseline_key)

print("Baseline uploaded to:", s3_uri(S3_BUCKET, baseline_key))

# ------------------------------------
# Generate baseline statistics (SageMaker Monitor)
# ------------------------------------
monitor = DefaultModelMonitor(
    role=EXECUTION_ROLE_ARN,
    instance_count=1,
    instance_type="ml.t3.medium",
    volume_size_in_gb=30,
    max_runtime_in_seconds=3600,
    sagemaker_session=sagemaker_session,
)

try:
    monitor.suggest_baseline(
        baseline_dataset=s3_uri(S3_BUCKET, S3_TRAIN_KEY),
        dataset_format=DatasetFormat.csv(header=True),
        output_s3_uri=s3_uri(S3_BUCKET, S3_MONITOR_BASELINE_PREFIX),
    )
    print("Baseline statistics job started.")
except Exception as e:
    print("Failed baseline job:", e)

# ------------------------------------
# Induce drift
# ------------------------------------
print("Inducing drift...")

df_drift = df_test.copy()
df_drift["passenger_count"] = df_drift["passenger_count"].sample(frac=1).reset_index(drop=True)
df_drift["distance_km"] *= np.random.uniform(1.2, 1.8, size=len(df_drift))

X_drift = df_drift[required_features].copy()
for c in numeric_cols:
    X_drift[c] = pd.to_numeric(X_drift[c], errors="coerce")

X_drift["pickup_datetime"] = X_drift["pickup_datetime"].astype(str)
X_drift = X_drift.fillna(0)

drift_preds = predict_batch_runtime(X_drift)
df_drift["prediction"] = drift_preds
df_drift["target"] = df_drift[target_col]

rmse_drift = mean_squared_error(df_drift["target"], df_drift["prediction"]) ** 0.5
print(f"DRIFT RMSE: {rmse_drift:.4f} (baseline: {rmse_baseline:.4f})")

# ------------------------------------
# Upload drift dataset
# ------------------------------------
drift_local = "/tmp/drift.csv"
df_drift.to_csv(drift_local, index=False)

drift_key = f"{S3_MONITOR_OUTPUT_PREFIX}/drift_{int(time.time())}.csv"
s3.upload_file(drift_local, S3_BUCKET, drift_key)

print("Drift dataset uploaded:", s3_uri(S3_BUCKET, drift_key))

# ------------------------------------
# Generate Evidently Report (Option A)
# ------------------------------------
print("\nBuilding Evidently report...")

timestamp = datetime.now().strftime("%Y%m%d-%H%M")

report = Report(metrics=[
    DataDriftPreset(),
    RegressionPreset(),
    RegressionQualityMetric(),
])

report.run(reference_data=df_baseline, current_data=df_drift)

html_local = f"/tmp/evidently_report_{timestamp}.html"
json_local = f"/tmp/evidently_report_{timestamp}.json"

report.save_html(html_local)

with open(json_local, "w") as f:
    json.dump(report.as_dict(), f, indent=4)

# Upload both
html_key = f"{S3_REPORT_PREFIX}/evidently_report_{timestamp}.html"
json_key = f"{S3_REPORT_PREFIX}/evidently_report_{timestamp}.json"

s3.upload_file(html_local, S3_BUCKET, html_key)
s3.upload_file(json_local, S3_BUCKET, json_key)

print("Evidently HTML saved to:", s3_uri(S3_BUCKET, html_key))
print("Evidently JSON saved to:", s3_uri(S3_BUCKET, json_key))

print("\n🎉 Monitoring pipeline completed successfully!")
