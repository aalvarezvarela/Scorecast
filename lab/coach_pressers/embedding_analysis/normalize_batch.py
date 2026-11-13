"""Local preparation and explicit submission/collection of presser batches.

Invoked through ``normalize --mode prepare|submit|status|collect``. Preparation
needs only the stored analysis artifacts. The other modes use the OpenAI key
via SETTINGS. Collection joins results by custom_id, never by output order.
It does not retry failed requests or fall back to synchronous billing.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from lab.coach_pressers.embedding_analysis.normalize import (
    REASONING_EFFORT,
    default_client,
    request_hash,
    response_report,
    save_report,
)

ENDPOINT = "/v1/responses"
TERMINAL = {"completed", "expired", "cancelled", "failed"}


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def prepare(batch_dir: Path, output_dir: Path, jobs: list[dict]) -> None:
    """Write the JSONL and its local manifest, without constructing a client."""
    if batch_dir.exists() and any(batch_dir.iterdir()):
        raise FileExistsError(f"Batch directory is not empty: {batch_dir}; choose --batch-dir")
    if not jobs or len(jobs) > 50_000:
        raise ValueError("A batch must contain 1 to 50,000 requests")
    records, entries = [], {}
    for job in jobs:
        presser_id = job["presser_id"]
        if presser_id in entries:
            raise ValueError(f"Duplicate presser_id: {presser_id}")
        records.append(json.dumps({
            "custom_id": presser_id, "method": "POST", "url": ENDPOINT,
            "body": job["body"],
        }, ensure_ascii=False))
        entries[presser_id] = {
            key: job[key] for key in ("team", "opponent", "request_hash")
        }
    payload = ("\n".join(records) + "\n").encode("utf-8")
    if len(payload) > 200_000_000:
        raise ValueError("Batch input exceeds 200 MB; prepare smaller batches")
    batch_dir.mkdir(parents=True, exist_ok=True)
    (batch_dir / "requests.jsonl").write_bytes(payload)
    write_json(batch_dir / "manifest.json", {
        "output_dir": str(output_dir.resolve()),
        "model": jobs[0]["body"]["model"],
        "reasoning_effort": REASONING_EFFORT,
        "entries": entries,
    })
    print(f"Prepared {len(entries)} requests in {batch_dir}; nothing sent to OpenAI")


def validate_input(batch_dir: Path, manifest: dict) -> None:
    """Keep the input and its local IDs/settings tied together before upload."""
    seen = set()
    for line in (batch_dir / "requests.jsonl").read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        presser_id = record["custom_id"]
        if presser_id in seen or presser_id not in manifest["entries"]:
            raise ValueError(f"Duplicate or unexpected custom_id: {presser_id}")
        seen.add(presser_id)
        if (
            record["url"] != ENDPOINT or record["method"] != "POST"
            or record["body"]["model"] != manifest["model"]
            or request_hash(record["body"]) != manifest["entries"][presser_id]["request_hash"]
        ):
            raise ValueError(f"Batch input no longer matches manifest: {presser_id}")
    if seen != set(manifest["entries"]):
        raise ValueError("Batch input is missing manifest entries")


def submit(batch_dir: Path) -> None:
    """Upload once, submit a 24h batch and persist its ID for later commands."""
    manifest = read_json(batch_dir / "manifest.json")
    batch_file = batch_dir / "batch.json"
    if batch_file.exists():
        saved = read_json(batch_file)
        print(f"Already submitted: {saved['id']}; use --mode status or collect")
        return
    validate_input(batch_dir, manifest)
    client = default_client()
    upload_file = batch_dir / "upload.json"
    if upload_file.exists():
        upload = read_json(upload_file)
    else:
        with (batch_dir / "requests.jsonl").open("rb") as handle:
            upload = client.files.create(file=handle, purpose="batch").model_dump()
        write_json(upload_file, upload)
    result = client.batches.create(
        input_file_id=upload["id"], endpoint=ENDPOINT, completion_window="24h",
        metadata={"task": "coach_presser_normalization"},
    ).model_dump()
    write_json(batch_file, result)
    print(f"Submitted {result['id']}: {result['status']}")


def refresh(batch_dir: Path, client) -> dict:
    saved = read_json(batch_dir / "batch.json")
    result = client.batches.retrieve(saved["id"]).model_dump()
    write_json(batch_dir / "batch.json", result)
    print(f"{result['id']}: {result['status']}; requests: {result.get('request_counts')}")
    if result.get("errors"):
        print(json.dumps(result["errors"], ensure_ascii=False))
    return result


def status(batch_dir: Path) -> None:
    refresh(batch_dir, default_client())


def watch(batch_dir: Path) -> None:
    """Poll for up to 25 hours and collect automatically; never submit work."""
    from openai import APIConnectionError, APIStatusError

    client = default_client()
    deadline = time.monotonic() + 25 * 3600
    while time.monotonic() < deadline:
        try:
            result = refresh(batch_dir, client)
            if result["status"] in TERMINAL:
                collect(batch_dir)
                return
        except APIConnectionError as exc:
            print(f"Connection error ({type(exc).__name__}); retrying in 60 seconds", flush=True)
        except APIStatusError as exc:
            if exc.status_code not in (408, 409, 429) and exc.status_code < 500:
                raise
            print(f"HTTP {exc.status_code}; retrying in 60 seconds", flush=True)
        time.sleep(60)
    raise TimeoutError("Batch watcher reached 25 hours; use --mode collect later")


def collect(batch_dir: Path) -> None:
    """Save valid results, and record per-request errors including missing IDs."""
    manifest = read_json(batch_dir / "manifest.json")
    client = default_client()
    result = refresh(batch_dir, client)
    if result["status"] not in TERMINAL:
        print("Batch still running; collect again later")
        return
    errors, warnings, saved, seen = {}, {}, [], set()
    output_dir = Path(manifest["output_dir"])
    for kind in ("output", "error"):
        file_id = result.get(f"{kind}_file_id")
        if not file_id:
            continue
        path = batch_dir / f"{kind}.jsonl"
        path.write_bytes(client.files.content(file_id).content)
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            presser_id = record.get("custom_id")
            if presser_id not in manifest["entries"] or presser_id in seen:
                raise ValueError(f"Unexpected or duplicate result custom_id: {presser_id}")
            seen.add(presser_id)
            response = record.get("response") or {}
            if record.get("error") or response.get("status_code") != 200:
                errors[presser_id] = record.get("error") or response
                continue
            body = response.get("body") or {}
            try:
                report = response_report(body)
            except (ValueError, KeyError, TypeError) as exc:
                errors[presser_id] = str(exc)
                continue
            metadata = {
                **manifest["entries"][presser_id],
                "model": manifest["model"],
                "reasoning_effort": manifest["reasoning_effort"],
                "mode": "batch", "batch_id": result["id"],
                "response_id": body.get("id"), "usage": body.get("usage"),
            }
            save_report(output_dir, presser_id, report, metadata)
            checks = read_json(output_dir / f"{presser_id}.meta.json")["problems"]
            if checks:
                warnings[presser_id] = checks
            saved.append(presser_id)
    for presser_id in set(manifest["entries"]) - seen:
        errors[presser_id] = f"No result returned; batch status: {result['status']}"
    write_json(batch_dir / "collection.json", {
        "batch_id": result["id"], "status": result["status"],
        "saved": saved, "errors": errors, "warnings": warnings,
    })
    print(f"Saved {len(saved)} reports; {len(errors)} errors; {len(warnings)} with rule warnings")
