"""Polite HTTP client for the injury-report CDN, with the false-403 defence.

The CDN answers a missing file with ``403 AccessDenied`` (S3 without
``ListBucket``) -- there is no 404. It *also* answers ``403`` when it decides you
have asked for too much, with no ``429`` and no ``Retry-After``. The two are
byte-identical, so a scraper that reads 403 as "missing" will quietly write
thousands of real reports off as absent. During research a 4-worker sweep did
exactly that: ~2,200 consecutive false 403s over a period that was fully
populated.

The defence is a **canary**: a URL verified to exist. On any 403 we ask the
canary. Canary 200 means the 403 was real. Canary 403 means we are throttled --
so we cool down, and if that does not clear it we stop the run rather than record
a single negative result.

Measured behaviour that sets the defaults: sequential requests at ~2.8 req/s ran
3,000 consecutive calls with zero 403s, and recovery from a trip took 20-120 s.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from enum import Enum

import requests
from tqdm.auto import tqdm

USER_AGENT = "nba-ou-injury-archiver/0.1 (+https://github.com/panchojasen; research)"

#: Verified to exist; used only to tell a real 403 from a throttled one.
CANARY_URL = (
    "https://ak-static.cms.nba.com/referee/injury/Injury-Report_2023-11-15_08PM.pdf"
)

# Deliberately well below what the CDN tolerates. Measured: 3,000 sequential
# requests at 2.78 req/s returned zero 403s, and the one run that ever tripped
# used 4 parallel workers at ~8 req/s. These defaults give ~1.2 req/s -- roughly
# half the rate already shown to be safe, single-threaded. The backfill is
# resumable, so the cost of being slow is wall-clock we do not have to sit and
# watch; the cost of being fast is corrupting the manifest with false absences.
DEFAULT_DELAY_S = 0.60
DEFAULT_JITTER_S = 0.20
DEFAULT_TIMEOUT_S = 20.0
# Measured recovery from a trip was 20-120 s; 180 s leaves margin.
DEFAULT_COOLDOWN_S = 180.0
DEFAULT_MAX_COOLDOWNS = 4
DEFAULT_RETRIES = 4


class Verdict(Enum):
    EXISTS = "exists"
    MISSING = "missing"
    UNKNOWN = "unknown"


class ThrottledError(RuntimeError):
    """The canary stayed down across every cooldown; stop and checkpoint."""


@dataclass
class Probe:
    verdict: Verdict
    status: int | None = None
    content_type: str = ""
    content_length: int | None = None
    etag: str = ""
    last_modified: str = ""
    note: str = ""


class ArchiveClient:
    def __init__(
        self,
        *,
        delay_s: float = DEFAULT_DELAY_S,
        jitter_s: float = DEFAULT_JITTER_S,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        cooldown_s: float = DEFAULT_COOLDOWN_S,
        max_cooldowns: int = DEFAULT_MAX_COOLDOWNS,
        retries: int = DEFAULT_RETRIES,
        canary_url: str = CANARY_URL,
        verbose: bool = True,
    ) -> None:
        self.delay_s = delay_s
        self.jitter_s = jitter_s
        self.timeout_s = timeout_s
        self.cooldown_s = cooldown_s
        self.max_cooldowns = max_cooldowns
        self.retries = retries
        self.canary_url = canary_url
        self.verbose = verbose
        self.session = self._new_session()
        self.requests_made = 0
        self.cooldowns_taken = 0

    def _new_session(self) -> requests.Session:
        s = requests.Session()
        s.headers.update({"User-Agent": USER_AGENT, "Accept": "application/pdf,*/*"})
        s.mount(
            "https://",
            requests.adapters.HTTPAdapter(
                pool_connections=2, pool_maxsize=2, max_retries=0
            ),
        )
        return s

    def _sleep_politely(self) -> None:
        time.sleep(self.delay_s + random.uniform(0.0, self.jitter_s))

    def _log(self, msg: str) -> None:
        # tqdm.write keeps these from shredding an active progress bar, and
        # behaves like print() when there is no bar.
        if self.verbose:
            tqdm.write(msg)

    def _canary_is_up(self) -> bool:
        try:
            r = self.session.head(self.canary_url, timeout=self.timeout_s)
            return r.status_code == 200
        except requests.RequestException:
            return False

    def _wait_out_throttle(self) -> None:
        """Cool down until the canary recovers, or give up on the run."""
        for attempt in range(1, self.max_cooldowns + 1):
            self._log(
                f"  ! throttled (canary 403) - cooling down "
                f"{self.cooldown_s:.0f}s [{attempt}/{self.max_cooldowns}]"
            )
            time.sleep(self.cooldown_s)
            self.cooldowns_taken += 1
            self.session = self._new_session()
            if self._canary_is_up():
                self._log("  + canary recovered, resuming")
                return
        raise ThrottledError(
            f"canary still 403 after {self.max_cooldowns} cooldowns; "
            "stopping so nothing is recorded as missing"
        )

    def _request(self, method: str, url: str, **kwargs) -> requests.Response | None:
        last: Exception | None = None
        for attempt in range(self.retries):
            if attempt:
                time.sleep(2.0**attempt)
            try:
                self.requests_made += 1
                return self.session.request(
                    method, url, timeout=self.timeout_s, **kwargs
                )
            except requests.RequestException as exc:
                last = exc
        self._log(f"  ! network error after {self.retries} attempts: {last}")
        return None

    def probe(self, url: str) -> Probe:
        """HEAD ``url`` and classify the answer, resolving 403 via the canary."""
        self._sleep_politely()
        resp = self._request("HEAD", url)
        if resp is None:
            return Probe(Verdict.UNKNOWN, note="network_error")

        if resp.status_code == 200:
            return Probe(
                Verdict.EXISTS,
                status=200,
                content_type=resp.headers.get("Content-Type", ""),
                content_length=int(resp.headers.get("Content-Length") or 0) or None,
                etag=resp.headers.get("ETag", "").strip('"'),
                last_modified=resp.headers.get("Last-Modified", ""),
            )

        if resp.status_code == 403:
            # The ambiguous case: ask the canary who is right.
            if self._canary_is_up():
                return Probe(Verdict.MISSING, status=403)
            self._wait_out_throttle()
            retry = self._request("HEAD", url)
            if retry is not None and retry.status_code == 200:
                return Probe(
                    Verdict.EXISTS,
                    status=200,
                    content_type=retry.headers.get("Content-Type", ""),
                    content_length=int(retry.headers.get("Content-Length") or 0)
                    or None,
                    etag=retry.headers.get("ETag", "").strip('"'),
                    last_modified=retry.headers.get("Last-Modified", ""),
                )
            if retry is not None and retry.status_code == 403:
                return Probe(Verdict.MISSING, status=403, note="after_cooldown")
            return Probe(Verdict.UNKNOWN, status=getattr(retry, "status_code", None))

        if resp.status_code == 429:
            self._wait_out_throttle()
            return Probe(Verdict.UNKNOWN, status=429, note="rate_limited")

        return Probe(Verdict.UNKNOWN, status=resp.status_code, note="unexpected_status")

    def fetch(self, url: str) -> tuple[bytes | None, Probe]:
        """GET ``url``; returns ``(body, probe)`` with the same 403 semantics."""
        self._sleep_politely()
        resp = self._request("GET", url)
        if resp is None:
            return None, Probe(Verdict.UNKNOWN, note="network_error")

        if resp.status_code == 200:
            return resp.content, Probe(
                Verdict.EXISTS,
                status=200,
                content_type=resp.headers.get("Content-Type", ""),
                content_length=len(resp.content),
                etag=resp.headers.get("ETag", "").strip('"'),
                last_modified=resp.headers.get("Last-Modified", ""),
            )

        if resp.status_code == 403:
            if self._canary_is_up():
                return None, Probe(Verdict.MISSING, status=403)
            self._wait_out_throttle()
            retry = self._request("GET", url)
            if retry is not None and retry.status_code == 200:
                return retry.content, Probe(
                    Verdict.EXISTS, status=200, content_length=len(retry.content)
                )
            return None, Probe(Verdict.UNKNOWN, status=403, note="after_cooldown")

        return None, Probe(Verdict.UNKNOWN, status=resp.status_code)
