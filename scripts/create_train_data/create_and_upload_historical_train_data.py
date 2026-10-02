#!/usr/bin/env python3
"""
Create historical training dataset (2 years before today) and upload to S3.

This script:
1. Calculates the date 30 days before today
2. Calls `create_df_to_predict` to generate training data up to that date
3. Saves the file locally
4. Uploads the file to S3 in the train_data folder
"""

import os
from datetime import datetime, timedelta
from io import BytesIO

from nba_ou.config.dataset_versions import TRAINING_DATA_SCHEMA_VERSION
from nba_ou.config.settings import SETTINGS
from nba_ou.create_training_data.create_df_to_predict import create_df_to_predict
from nba_ou.create_training_data.train_data_store import (
    historical_filename,
    train_data_key,
)
from nba_ou.utils.s3_models import make_s3_client, upload_bytes_to_s3


def main() -> None:
    """Create historical training data (30 days before today) and upload to S3."""

    # Calculate the date 30 days before today
    today = datetime.now()
    thirty_days_ago = today - timedelta(
        days=2 * 30
    )  # Approximate, can adjust for leap years
    limit_date = thirty_days_ago.strftime("%Y-%m-%d")

    print(f"Creating training data up to: {limit_date}")
    print(f"(30 days before today: {thirty_days_ago.strftime('%Y-%m-%d')})")

    # Call create_df_to_predict without a scheduled date (no todays prediction)
    df_train = create_df_to_predict(
        todays_prediction=False,
        recent_limit_to_include=limit_date,
        older_season_limit=None,  # Uses default (all from 2017-18)
    )

    print(f"Training data created. Shape: {df_train.shape}")

    # The schema version goes in the name and in the prefix. This script and
    # create_train_data.py call the same create_df_to_predict, so the frame
    # landing in S3 has exactly the schema that script stamps onto its CSVs --
    # it was simply not being recorded here.
    filename = historical_filename(
        schema_version=TRAINING_DATA_SCHEMA_VERSION,
        limit_date=thirty_days_ago.strftime("%Y%m%d"),
    )

    # Convert object columns to string to avoid Parquet type errors
    for col in df_train.select_dtypes(include=["object"]).columns:
        df_train[col] = df_train[col].astype(str)

    # Convert DataFrame to Parquet bytes (no local file)
    buffer = BytesIO()
    df_train.to_parquet(buffer, index=False)
    parquet_bytes = buffer.getvalue()
    print(f"Parquet data prepared ({len(parquet_bytes):,} bytes)")

    # Upload to S3
    print("\nUploading to S3...")
    region = os.getenv("S3_AWS_REGION") or SETTINGS.s3_aws_region
    bucket = os.getenv("S3_BUCKET") or SETTINGS.s3_bucket
    profile = SETTINGS.s3_aws_profile

    # S3 key: train_data/<schema_version>/<filename>
    s3_key = train_data_key(
        schema_version=TRAINING_DATA_SCHEMA_VERSION, filename=filename
    )

    print(f"  Bucket: {bucket}")
    print(f"  Key: {s3_key}")
    print(f"  Region: {region}")
    print(f"  Profile: {profile or '<none>'}")

    s3_client = make_s3_client(profile=profile, region=region)
    upload_bytes_to_s3(
        s3_client=s3_client,
        bucket=bucket,
        key=s3_key,
        data=parquet_bytes,
    )

    print(f"✅ Successfully uploaded to S3: s3://{bucket}/{s3_key}")


if __name__ == "__main__":
    main()
