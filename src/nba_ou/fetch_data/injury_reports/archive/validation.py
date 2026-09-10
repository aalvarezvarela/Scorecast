"""Is this actually the report we asked for?

The strongest check available is nearly free: every report's first page starts
``Injury Report: MM/DD/YY hh:mm AM/PM``, so the file states its own publication
time. Comparing that against the time we *derived* from the filename validates
the era model, the label decoding and the DST handling in one step -- and it is
what would catch a future era where ``_08PM`` starts meaning something new.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import datetime

import pymupdf

MIN_PDF_BYTES = 2048
PDF_MAGIC = b"%PDF"

_HEADER_RE = re.compile(
    r"Injury\s+Report:\s*(\d{2})/(\d{2})/(\d{2})\s+(\d{2}):(\d{2})\s*(AM|PM)",
    re.IGNORECASE,
)

OK = "ok"
NO_MAGIC = "pdf_magic_missing"
TOO_SMALL = "too_small"
UNOPENABLE = "unopenable"
NO_HEADER = "header_not_found"
MISMATCH = "date_mismatch"
IMAGE_ONLY = "image_only"


@dataclass
class Validation:
    status: str
    sha256: str
    page_count: int | None = None
    header_datetime_et: datetime | None = None
    detail: str = ""

    @property
    def ok(self) -> bool:
        # image_only is a known, accepted 2018 condition, not a failure.
        return self.status in (OK, IMAGE_ONLY)


def parse_header_datetime(text: str) -> datetime | None:
    m = _HEADER_RE.search(text)
    if not m:
        return None
    mm, dd, yy, hh, mi, ap = m.groups()
    hour = int(hh) % 12 + (12 if ap.upper() == "PM" else 0)
    return datetime(2000 + int(yy), int(mm), int(dd), hour, int(mi))


def validate(data: bytes, expected_et: datetime) -> Validation:
    sha = hashlib.sha256(data).hexdigest()

    if not data.startswith(PDF_MAGIC):
        return Validation(NO_MAGIC, sha, detail=repr(data[:16]))
    if len(data) < MIN_PDF_BYTES:
        return Validation(TOO_SMALL, sha, detail=f"{len(data)} bytes")

    try:
        doc = pymupdf.open(stream=data, filetype="pdf")
        pages = doc.page_count
        text = "".join(p.get_text() for p in doc)
    except Exception as exc:  # noqa: BLE001 - any parse failure is a failure
        return Validation(UNOPENABLE, sha, detail=str(exc)[:120])

    if len(text.strip()) < 200:
        # The Oct-Nov 2018 reports are scanned images; keep them, flag them.
        return Validation(IMAGE_ONLY, sha, page_count=pages)

    header = parse_header_datetime(text)
    if header is None:
        return Validation(NO_HEADER, sha, page_count=pages)

    naive_expected = expected_et.replace(tzinfo=None)
    if header != naive_expected:
        return Validation(
            MISMATCH,
            sha,
            page_count=pages,
            header_datetime_et=header,
            detail=f"header={header.isoformat()} expected={naive_expected.isoformat()}",
        )

    return Validation(OK, sha, page_count=pages, header_datetime_et=header)
