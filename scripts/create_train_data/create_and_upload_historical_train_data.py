#!/usr/bin/env python3
"""
Create historical training dataset (2 years before today) and upload to S3.

This script:
1. Calculates the date 30 days before today
2. Calls `create_df_to_predict` to generate training data up to that date
3. Adds every newer schema version's columns as layers on that base frame
4. Uploads one Parquet file per version to train_data/<version>/ in S3
"""

import os
from datetime import datetime, timedelta
from io import BytesIO

from nba_ou.config.dataset_versions import BASE_SCHEMA_VERSION
from nba_ou.config.settings import SETTINGS
from nba_ou.create_training_data.create_df_to_predict import create_df_to_predict
from nba_ou.create_training_data.schema_layers import (
    CLOSING_LINE,
    LayerContext,
    apply_layers,
    available_versions,
)
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

    region = os.getenv("S3_AWS_REGION") or SETTINGS.s3_aws_region
    bucket = os.getenv("S3_BUCKET") or SETTINGS.s3_bucket
    profile = SETTINGS.s3_aws_profile
    s3_client = make_s3_client(profile=profile, region=region)
    print(f"  Bucket: {bucket}  Region: {region}  Profile: {profile or '<none>'}")

    # create_df_to_predict builds the base schema. Every newer version only adds
    # columns, so each is the previous frame plus its layer, uploaded under its
    # own version in the name and the prefix (train_data/<schema_version>/).
    ctx = LayerContext()
    previous = BASE_SCHEMA_VERSION
    for version in available_versions():
        if version != BASE_SCHEMA_VERSION:
            df_train = apply_layers(
                df_train,
                from_version=previous,
                to_version=version,
                dataset_type=CLOSING_LINE,
                ctx=ctx,
            )
        previous = version
        _upload(
            df_train,
            s3_client=s3_client,
            bucket=bucket,
            schema_version=version,
            limit_date=thirty_days_ago.strftime("%Y%m%d"),
        )


def _upload(
    df, *, s3_client, bucket: str, schema_version: str, limit_date: str
) -> None:
    filename = historical_filename(schema_version=schema_version, limit_date=limit_date)
    frame = df.copy()
    # Convert object columns to string to avoid Parquet type errors
    for col in frame.select_dtypes(include=["object"]).columns:
        frame[col] = frame[col].astype(str)

    # Convert DataFrame to Parquet bytes (no local file)
    buffer = BytesIO()
    frame.to_parquet(buffer, index=False)
    parquet_bytes = buffer.getvalue()
    s3_key = train_data_key(schema_version=schema_version, filename=filename)
    print(
        f"\nUploading {schema_version} ({len(parquet_bytes):,} bytes) to {s3_key} ..."
    )
    upload_bytes_to_s3(
        s3_client=s3_client,
        bucket=bucket,
        key=s3_key,
        data=parquet_bytes,
    )
    print(f"✅ Successfully uploaded to S3: s3://{bucket}/{s3_key}")


if __name__ == "__main__":
    main()
