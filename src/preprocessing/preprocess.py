import boto3
import pandas as pd
import io
import math
import pathlib
from botocore.exceptions import NoCredentialsError

BUCKET = "nyc-taxi-mlops-final-project"
INPUT_KEY = "processed/full_dataset.csv"
TRAIN_OUTPUT_KEY = "processed/train_dataset.csv"
TEST_OUTPUT_KEY = "processed/test_dataset.csv"


def haversine_distance(lat1, lon1, lat2, lon2):
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = (
        math.sin(dlat / 2)**2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    )
    c = 2 * math.asin(math.sqrt(a))
    return 6371 * c  # km


def add_features(df):
    df["pickup_datetime"] = pd.to_datetime(
        df["pickup_datetime"], errors="coerce")
    df = df.dropna(subset=["pickup_datetime"])

    df["hour"] = df["pickup_datetime"].dt.hour
    df["month"] = df["pickup_datetime"].dt.month
    df["weekday"] = df["pickup_datetime"].dt.weekday

    df["distance_km"] = df.apply(
        lambda r: haversine_distance(
            r["pickup_latitude"], r["pickup_longitude"],
            r["dropoff_latitude"], r["dropoff_longitude"]
        ),
        axis=1
    )

    return df


def clean_data(df):
    df = df[
        (df["pickup_latitude"].between(40, 42)) &
        (df["dropoff_latitude"].between(40, 42)) &
        (df["pickup_longitude"].between(-75, -72)) &
        (df["dropoff_longitude"].between(-75, -72))
    ]

    df = df[
        (df["fare_amount"] > 0) &
        (df["fare_amount"] < 200) &
        (df["passenger_count"] > 0) &
        (df["passenger_count"] <= 6) &
        (df["distance_km"] > 0) &
        (df["distance_km"] < 100)
    ]

    return df


def save_df(df, s3_client, bucket, key, is_s3):
    """Helper to save dataframe to S3 or Local"""
    if is_s3:
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        s3_client.put_object(Bucket=bucket, Key=key,
                             Body=csv_buffer.getvalue().encode("utf-8"))
        print(f"Uploaded: s3://{bucket}/{key}")
    else:
        repo_root = pathlib.Path(__file__).resolve().parents[2]
        local_path = repo_root / "data" / key
        local_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(local_path, index=False)
        print(f"Saved locally: {local_path}")


def main():
    s3 = boto3.client("s3")

    # Try S3 first; if credentials are missing, fall back to a local dataset
    use_s3 = True
    print("Downloading dataset...")
    try:
        obj = s3.get_object(Bucket=BUCKET, Key=INPUT_KEY)
        df = pd.read_csv(io.BytesIO(obj["Body"].read()))
    except NoCredentialsError:
        print("AWS credentials not found. Loading local dataset instead.")
        use_s3 = False
        repo_root = pathlib.Path(__file__).resolve().parents[2]
        local_path = repo_root / "data" / INPUT_KEY
        if local_path.exists():
            df = pd.read_csv(local_path)
            print(f"Loaded local dataset: {local_path}")
        else:
            raise RuntimeError(
                f"No AWS credentials and local dataset not found at {local_path}.")
    except Exception as e:
        print("Failed to download from S3:", e)
        repo_root = pathlib.Path(__file__).resolve().parents[2]
        local_path = repo_root / "data" / INPUT_KEY
        if local_path.exists():
            print("Falling back to local dataset.")
            df = pd.read_csv(local_path)
            use_s3 = False
        else:
            raise

    print("Starting preprocessing...")
    
    # 1. Feature Engineering (converts datetime)
    df = add_features(df)
    
    # 2. Cleaning (removes invalid rows)
    df = clean_data(df)

    # 3. Sort by date (Ascending)
    df = df.sort_values(by="pickup_datetime", ascending=True)

    # 4. Split into Train (800) and Test (200)
    # We take the first 1000 rows total.
    # If the clean dataset is smaller than 1000, this will just take what is available.
    
    train_df = df.iloc[:800]
    test_df = df.iloc[800:1000]

    print(f"Train shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")

    # Save Train
    save_df(train_df, s3, BUCKET, TRAIN_OUTPUT_KEY, use_s3)
    
    # Save Test
    save_df(test_df, s3, BUCKET, TEST_OUTPUT_KEY, use_s3)


if __name__ == "__main__":
    main()