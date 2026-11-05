"""An in-memory S3 with conditional writes, for the resumable season archives.

Covers what ``lab.game_reports.season_reports`` and the recap audio/transcript
archives call: ``put_object`` honouring ``IfNoneMatch="*"`` and ``IfMatch``,
``get_object``, ``head_object`` and the ``list_objects_v2`` paginator. Each
stored object keeps every ``put_object`` argument, so tests can assert on
headers such as ``ContentType``. Keys in ``fail_keys`` raise ``AccessDenied``.
"""

from __future__ import annotations

import hashlib
import io

from botocore.exceptions import ClientError


def client_error(code: str) -> ClientError:
    return ClientError({"Error": {"Code": code, "Message": code}}, "test")


class ConditionalFakeS3:
    def __init__(self) -> None:
        self.objects: dict[str, dict] = {}
        self.fail_keys: set[str] = set()
        self.puts: list[str] = []

    def put_object(self, **kwargs):
        key = kwargs["Key"]
        if key in self.fail_keys:
            raise client_error("AccessDenied")
        previous = self.objects.get(key)
        if kwargs.get("IfNoneMatch") == "*" and previous:
            raise client_error("PreconditionFailed")
        if "IfMatch" in kwargs and (
            not previous or previous["ETag"] != kwargs["IfMatch"]
        ):
            raise client_error("PreconditionFailed")
        self.objects[key] = {
            "Metadata": {},
            **kwargs,
            "ETag": hashlib.sha256(kwargs["Body"]).hexdigest(),
        }
        self.puts.append(key)
        return {"ETag": self.objects[key]["ETag"]}

    def get_object(self, **kwargs):
        if kwargs["Key"] not in self.objects:
            raise client_error("NoSuchKey")
        stored = self.objects[kwargs["Key"]]
        return {**stored, "Body": io.BytesIO(stored["Body"])}

    def head_object(self, **kwargs):
        stored = self.objects[kwargs["Key"]]
        return {"ContentLength": len(stored["Body"]), "Metadata": stored["Metadata"]}

    def get_paginator(self, name):
        return self

    def paginate(self, **kwargs):
        yield {
            "Contents": [
                {"Key": key, "Size": len(obj["Body"])}
                for key, obj in self.objects.items()
                if key.startswith(kwargs["Prefix"])
            ]
        }
