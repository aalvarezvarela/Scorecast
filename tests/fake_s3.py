"""An in-memory stand-in for the subset of the S3 API the registry uses.

Only ``get_object``, ``put_object``, ``delete_object`` and the
``list_objects_v2`` paginator, because that is all
``nba_ou.utils.s3_models`` calls. Missing keys raise a real ``ClientError``
with ``NoSuchKey`` so the production code's error handling is exercised rather
than bypassed.
"""

from __future__ import annotations

import io

from botocore.exceptions import ClientError


class _Body:
    def __init__(self, data: bytes) -> None:
        self._stream = io.BytesIO(data)

    def read(self) -> bytes:
        return self._stream.read()


class _Paginator:
    def __init__(self, store: dict[str, bytes]) -> None:
        self._store = store

    def paginate(self, Bucket: str, Prefix: str = ""):  # noqa: N803 - boto3 casing
        contents = [
            {"Key": key, "Size": len(value)}
            for key, value in sorted(self._store.items())
            if key.startswith(Prefix)
        ]
        # Deliberately paged, so callers that forget to paginate are caught.
        page_size = 2
        if not contents:
            yield {}
            return
        for start in range(0, len(contents), page_size):
            yield {"Contents": contents[start : start + page_size]}


class FakeS3Client:
    """Records every write, so tests can assert on ordering as well as state."""

    def __init__(self) -> None:
        self.store: dict[str, bytes] = {}
        self.put_order: list[str] = []

    def get_object(self, Bucket: str, Key: str):  # noqa: N803
        if Key not in self.store:
            raise ClientError(
                {"Error": {"Code": "NoSuchKey", "Message": "Not found"}},
                "GetObject",
            )
        return {"Body": _Body(self.store[Key])}

    def put_object(self, Bucket: str, Key: str, Body: bytes):  # noqa: N803
        self.store[Key] = Body
        self.put_order.append(Key)
        return {}

    def delete_object(self, Bucket: str, Key: str):  # noqa: N803
        self.store.pop(Key, None)
        return {}

    def copy_object(self, Bucket: str, CopySource: dict, Key: str):  # noqa: N803
        self.store[Key] = self.store[CopySource["Key"]]
        self.put_order.append(Key)
        return {}

    def upload_file(self, filename: str, bucket: str, key: str):
        with open(filename, "rb") as handle:
            self.put_object(Bucket=bucket, Key=key, Body=handle.read())

    def get_paginator(self, name: str):
        assert name == "list_objects_v2"
        return _Paginator(self.store)
