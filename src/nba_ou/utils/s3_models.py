import io
import json

import boto3
import pandas as pd


def make_s3_client(*, profile: str | None, region: str, config=None):
    session_kwargs = {"region_name": region}
    if profile:
        session_kwargs["profile_name"] = profile
    session = boto3.Session(**session_kwargs)
    return session.client("s3", **({"config": config} if config is not None else {}))


def read_s3_object_bytes(*, s3_client, bucket: str, key: str) -> bytes:
    resp = s3_client.get_object(Bucket=bucket, Key=key)
    return resp["Body"].read()


def read_s3_json_object(*, s3_client, bucket: str, key: str) -> dict:
    return json.loads(read_s3_object_bytes(s3_client=s3_client, bucket=bucket, key=key))


def upload_bytes_to_s3(*, s3_client, bucket: str, key: str, data: bytes) -> None:
    """Upload bytes directly to S3.

    Args:
        s3_client: Boto3 S3 client
        bucket: S3 bucket name
        key: S3 object key (path within bucket)
        data: Bytes data to upload
    """
    s3_client.put_object(Bucket=bucket, Key=key, Body=data)


def list_s3_objects(*, s3_client, bucket: str, prefix: str) -> list[dict]:
    """
    List objects under an S3 prefix.

    Returns:
        list[dict]: Raw S3 object dictionaries (contains Key, LastModified, Size, ...)
    """
    paginator = s3_client.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=bucket, Prefix=prefix)

    objects: list[dict] = []
    for page in pages:
        objects.extend(page.get("Contents", []))
    return objects


def load_parquet_from_bytes(b: bytes) -> pd.DataFrame:
    """Load a parquet DataFrame from bytes."""
    return pd.read_parquet(io.BytesIO(b))
