# drift_check.py
# Improved drift detection script with stable numeric/categorical plots + drift metrics.

import subprocess
import sys
import argparse
import os
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Install required libs (only for Processing container)
packages = [
    "pandas==2.2.1",
    "numpy==1.25.2",
    "matplotlib==3.8.1",
    "seaborn==0.12.2",
    "scipy"    # for drift tests
]
for pkg in packages:
    subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])


from scipy.stats import ks_2samp, chisquare


# ----------------------------------------------------------
# Load CSV file or directory
# ----------------------------------------------------------
def load_csv(path):
    if os.path.isdir(path):
        files = [f for f in os.listdir(path) if f.endswith(".csv")]
        if not files:
            raise RuntimeError(f"No CSV in directory: {path}")
        path = os.path.join(path, files[0])

    if not os.path.exists(path):
        raise RuntimeError(f"File not found: {path}")

    return pd.read_csv(path)


# ----------------------------------------------------------
# NUMERIC PLOTS (KDE + fallback)
# ----------------------------------------------------------
def plot_numeric_feature(baseline_df, new_df, feature, output_dir):
    plt.figure(figsize=(8, 4))

    b = baseline_df[feature].dropna()
    n = new_df[feature].dropna()

    try:
        sns.kdeplot(b, label="Baseline", fill=True)
        sns.kdeplot(n, label="New", fill=True)
    except Exception:
        sns.histplot(b, label="Baseline", stat="density", kde=False, alpha=0.5)
        sns.histplot(n, label="New", stat="density", kde=False, alpha=0.5)

    plt.title(f"Distribution: {feature}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{feature}_distribution.png")
    plt.close()


# ----------------------------------------------------------
# CATEGORICAL PLOTS
# ----------------------------------------------------------
def plot_categorical_feature(baseline_df, new_df, feature, output_dir):
    b_counts = baseline_df[feature].value_counts()
    n_counts = new_df[feature].value_counts()

    categories = sorted(set(b_counts.index).union(set(n_counts.index)))

    df = pd.DataFrame({
        "Baseline": b_counts.reindex(categories, fill_value=0),
        "New": n_counts.reindex(categories, fill_value=0),
    })

    df.plot(kind="bar", figsize=(8, 4))
    plt.title(f"Counts: {feature}")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{feature}_counts.png")
    plt.close()


# ----------------------------------------------------------
# SIMPLE DRIFT METRICS (KS test + Chi-square)
# ----------------------------------------------------------
def compute_drift_metrics(baseline_df, new_df):
    metrics = {}

    numeric_cols = baseline_df.select_dtypes(include="number").columns
    cat_cols = baseline_df.select_dtypes(exclude="number").columns

    # KS test for numeric
    for col in numeric_cols:
        try:
            stat, p = ks_2samp(
                baseline_df[col].dropna(),
                new_df[col].dropna()
            )
            metrics[col] = {"type": "numeric", "ks_stat": float(stat), "p_value": float(p)}
        except Exception as e:
            metrics[col] = {"type": "numeric", "error": str(e)}

    # Chi-square for categorical
    for col in cat_cols:
        b_counts = baseline_df[col].value_counts()
        n_counts = new_df[col].value_counts()

        cats = sorted(set(b_counts.index).union(set(n_counts.index)))

        b = b_counts.reindex(cats, fill_value=0)
        n = n_counts.reindex(cats, fill_value=0)

        try:
            stat, p = chisquare(f_obs=n, f_exp=b + 0.001)
            metrics[col] = {"type": "categorical", "chi2_stat": float(stat), "p_value": float(p)}
        except Exception as e:
            metrics[col] = {"type": "categorical", "error": str(e)}

    return metrics


# ----------------------------------------------------------
# MAIN
# ----------------------------------------------------------
def main(args):
    print("Loading datasets...")
    baseline_df = load_csv(args.baseline_dataset)
    new_df = load_csv(args.input_dataset)

    output_dir = "/opt/ml/processing/output"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    numeric_cols = baseline_df.select_dtypes(include="number").columns
    cat_cols = baseline_df.select_dtypes(exclude="number").columns

    # ---- Generate Plots ----
    for c in numeric_cols:
        plot_numeric_feature(baseline_df, new_df, c, output_dir)

    for c in cat_cols:
        plot_categorical_feature(baseline_df, new_df, c, output_dir)

    # ---- Compute Drift Scores ----
    drift_metrics = compute_drift_metrics(baseline_df, new_df)
    pd.DataFrame(drift_metrics).T.to_csv(f"{output_dir}/drift_metrics.csv")
    print("Drift metrics saved.")

    # ---- Upload to S3 ----
    import boto3

    def upload_folder_to_s3(local_folder, bucket, prefix):
        s3 = boto3.client("s3")
        for root, dirs, files in os.walk(local_folder):
            for f in files:
                local_path = os.path.join(root, f)
                rel_path = os.path.relpath(local_path, local_folder)
                key = f"{prefix}/{rel_path}"
                s3.upload_file(local_path, bucket, key)
                print("Uploaded:", key)

    upload_folder_to_s3(
        local_folder="/opt/ml/processing/output",
        bucket="mlops-project-uoc",
        prefix="monitoring/output/drift_plots"
    )

    print("Drift check finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_dataset", type=str, required=True)
    parser.add_argument("--input_dataset", type=str, required=True)
    parser.add_argument("--analysis_config", type=str, required=False)
    main(parser.parse_args())
