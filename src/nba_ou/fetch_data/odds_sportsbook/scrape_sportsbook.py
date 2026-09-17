import asyncio
import json
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date
from typing import Any

import pandas as pd
import requests
from nba_ou.fetch_data.odds_sportsbook.process_money_line_data import (
    load_one_day_moneyline_csv,
)
from nba_ou.fetch_data.odds_sportsbook.process_spread_data import (
    load_one_day_spread_csv,
)
from nba_ou.fetch_data.odds_sportsbook.process_total_lines_data import (
    load_one_day_totals_csv,
)
from nba_ou.postgre_db.odds_sportsbook.process_sportsbook_data import merge_daily_frames
from playwright.async_api import Page, async_playwright
from playwright.async_api import TimeoutError as PlaywrightTimeoutError
from tqdm import tqdm

BASE_URL = "https://www.sportsbookreview.com"
TIMEOUT_MS = 10_000
SLEEP_MIN_S = 0.4
SLEEP_MAX_S = 1.1


@dataclass(frozen=True)
class TotalsScrapeResult:
    df: pd.DataFrame | None
    no_odds: bool


def season_year_for_date(d: date) -> int:
    # NBA season start year (e.g. Jan 2026 belongs to 2025-26 => 2025)
    return d.year if d.month >= 10 else (d.year - 1)


def build_sbr_totals_url(d: date) -> str:
    return (
        f"{BASE_URL}/betting-odds/nba-basketball/totals/full-game/?date={d.isoformat()}"
    )


def build_sbr_moneyline_url(d: date) -> str:
    return f"{BASE_URL}/betting-odds/nba-basketball/money-line/full-game/?date={d.isoformat()}"


def build_sbr_spread_url(d: date) -> str:
    return f"{BASE_URL}/betting-odds/nba-basketball/?date={d.isoformat()}"


async def random_sleep(min_s: float = SLEEP_MIN_S, max_s: float = SLEEP_MAX_S) -> None:
    await asyncio.sleep(random.uniform(min_s, max_s))


async def try_click_consent(page: Page) -> None:
    candidates = [
        "Accept all",
        "I agree",
        "Agree",
        "Accept",
        "Aceptar",
        "Aceptar todo",
        "Continue",
        "OK",
        "Got it",
    ]
    for label in candidates:
        try:
            await page.get_by_role("button", name=label).click(timeout=500)
            return
        except Exception:
            pass


async def has_no_odds_message(page: Page) -> bool:
    js = r"""
    () => {
      const norm = (s) => (s || "").replace(/\s+/g, " ").trim();
      const els = Array.from(document.querySelectorAll("div"));
      return els.some(el => norm(el.textContent) === "No odds available at this time for this league");
    }
    """
    try:
        return bool(await page.evaluate(js))
    except Exception:
        return False


def _slugify_book(book: str) -> str:
    s = book.strip().lower()
    out = []
    prev_us = False
    for ch in s:
        if ch.isalnum():
            out.append(ch)
            prev_us = False
        else:
            if not prev_us:
                out.append("_")
                prev_us = True
    slug = "".join(out).strip("_")
    return slug.replace("logo", "").strip("_")


async def extract_sbr_totals_full_game_rows(
    page: Page, page_date: date
) -> pd.DataFrame:
    js = r"""
    () => {
      const norm = (s) => (s || "").replace(/\s+/g, " ").trim();

      const section = document.querySelector("#section-nba");
      if (!section) return { books: [], rows: [] };

      const thead = section.querySelector("#thead-nba");
      let books = [];
      if (thead) {
        const imgs = Array.from(thead.querySelectorAll("a img[alt]"));
        books = imgs
          .map(img => norm(img.getAttribute("alt")))
          .map(t => t.replace(/\s+Logo$/i, ""))
          .filter(Boolean);
      }

      const tbody = section.querySelector("#tbody-nba");
      if (!tbody) return { books, rows: [] };

      const leftEids = Array.from(tbody.querySelectorAll("[data-horizontal-eid]"));

      const ascendToGameRoot = (node) => {
        let cur = node;
        while (cur && cur !== section) {
          const hasOdds = cur.querySelector && cur.querySelector('a[data-aatracker^="Odds Table - Odds Cell CTA"]');
          if (hasOdds) return cur;
          cur = cur.parentElement;
        }
        return node;
      };

      const parsePct = (txt) => {
        const m = String(txt || "").match(/(\d+(?:\.\d+)?)\s*%/);
        return m ? m[1] : null;
      };

      const parseLineAndPrice = (txt) => {
        const t = String(txt || "");
        const side = (t.match(/\b([OU])\b/) || [null, null])[1];
        const line = (t.match(/([0-9]+(?:\.[0-9]+)?)/) || [null, null])[1];
        const price = (t.match(/([+-]\d{2,4})/) || [null, null])[1];
        return { side: side || null, line: line || null, price: price || null };
      };

      const parseCellTwoRows = (cell) => {
        const out = [
          { line: null, price: null },
          { line: null, price: null }
        ];
        if (!cell) return out;

        const btns = Array.from(cell.querySelectorAll('span[role="button"]'));
        const texts = btns.map(b => norm(b.textContent)).filter(Boolean);

        for (let i = 0; i < Math.min(2, texts.length); i++) {
          const parsed = parseLineAndPrice(texts[i]);
          out[i] = { line: parsed.line, price: parsed.price };
        }
        return out;
      };

      const rows = [];

      for (const eidNode of leftEids) {
        const eidRaw = eidNode.getAttribute("data-horizontal-eid");
        const event_id = eidRaw ? Number(eidRaw) : null;

        const root = ascendToGameRoot(eidNode);

        const timeEl = root.querySelector('[data-vertical-sbid="time"] span');
        const start_time = timeEl ? norm(timeEl.textContent) : null;

        const matchupA = root.querySelector('a[href^="/scores/nba-basketball/matchup/"]');
        const matchup_url = matchupA ? matchupA.getAttribute("href") : null;

        const leftCol = root.querySelector("div.col-3");
        const teamSpans = leftCol ? Array.from(leftCol.querySelectorAll("span.fw-bolder")) : [];
        const teams = teamSpans.map(s => norm(s.textContent)).filter(Boolean).slice(0, 2);

        const scoreEls = leftCol ? Array.from(leftCol.querySelectorAll('div[class*="scores"] div')) : [];
        const scores = scoreEls.map(s => norm(s.textContent)).filter(Boolean).slice(0, 2);

        const consPctCol = root.querySelector('[data-vertical-sbid="-2"]');
        const pctSpans = consPctCol ? Array.from(consPctCol.querySelectorAll("span.opener")) : [];
        const cons_pcts = pctSpans.map(s => parsePct(norm(s.textContent))).slice(0, 2);

        const consOpenCol = root.querySelector('[data-vertical-sbid="-1"]');
        const openRowSpans = consOpenCol
          ? Array.from(consOpenCol.querySelectorAll('span[data-cy="odd-grid-opener-homepage"]'))
          : [];
        const openRows = openRowSpans.map(s => parseLineAndPrice(norm(s.textContent))).slice(0, 2);

        const cellDivs = Array.from(root.querySelectorAll('div[class*="OddsCells_numbersContainer"]'));

        const n = books.length ? Math.min(books.length, cellDivs.length) : cellDivs.length;
        const cells = cellDivs.slice(0, n).map(c => parseCellTwoRows(c));

        for (let r = 0; r < 2; r++) {
          const team_name = teams[r] || null;
          const score = scores[r] || null;

          const openerParsed = openRows[r] || { side: null, line: null, price: null };
          const total_side = openerParsed.side || (r === 0 ? "O" : "U");

          const row = {
            event_id,
            start_time,
            matchup_url,
            row_index: r,
            team_name,
            score,
            consensus_pct: cons_pcts[r] || null,
            consensus_opener_side: total_side,
            consensus_opener_line: openerParsed.line || null,
            consensus_opener_price: openerParsed.price || null,
          };

          for (let j = 0; j < n; j++) {
            const book = books[j] || `book_${j}`;
            const book_key = book;
            const two = cells[j] || [{line:null,price:null},{line:null,price:null}];
            const v = two[r] || { line: null, price: null };
            row[`book__${book_key}__line`] = v.line;
            row[`book__${book_key}__price`] = v.price;
          }

          rows.push(row);
        }
      }

      return { books, rows };
    }
    """

    payload = await page.evaluate(js)
    rows: list[dict[str, Any]] = (payload or {}).get("rows") or []
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df.insert(0, "date", page_date.isoformat())

    def to_float(series: pd.Series) -> pd.Series:
        return pd.to_numeric(
            series.astype("string").str.replace("−", "-", regex=False), errors="coerce"
        )

    df["consensus_pct"] = to_float(df.get("consensus_pct", pd.Series(dtype="string")))
    df["consensus_opener_line"] = to_float(
        df.get("consensus_opener_line", pd.Series(dtype="string"))
    )
    df["consensus_opener_price"] = to_float(
        df.get("consensus_opener_price", pd.Series(dtype="string"))
    )

    for c in df.columns:
        if c.endswith("__line") or c.endswith("__price"):
            df[c] = to_float(df[c])

    rename_map: dict[str, str] = {}
    for c in df.columns:
        if c.startswith("book__") and ("__line" in c or "__price" in c):
            parts = c.split("__")
            if len(parts) == 3:
                book_name = parts[1]
                suffix = parts[2]
                rename_map[c] = f"{_slugify_book(book_name)}_{suffix}"
    df = df.rename(columns=rename_map)

    return df


async def extract_sbr_moneyline_full_game_rows(
    page: Page, page_date: date
) -> pd.DataFrame:
    js = r"""
    () => {
      const norm = (s) => (s || "").replace(/\s+/g, " ").trim();

      const section = document.querySelector("#section-nba");
      if (!section) return { books: [], rows: [] };

      const thead = section.querySelector("#thead-nba");
      let books = [];
      if (thead) {
        const imgs = Array.from(thead.querySelectorAll("a img[alt]"));
        books = imgs
          .map(img => norm(img.getAttribute("alt")))
          .map(t => t.replace(/\s+Logo$/i, ""))
          .filter(Boolean);
      }

      const tbody = section.querySelector("#tbody-nba");
      if (!tbody) return { books, rows: [] };

      const leftEids = Array.from(tbody.querySelectorAll("[data-horizontal-eid]"));

      const ascendToGameRoot = (node) => {
        let cur = node;
        while (cur && cur !== section) {
          const hasOdds = cur.querySelector && cur.querySelector('a[data-aatracker^="Odds Table - Odds Cell CTA"]');
          if (hasOdds) return cur;
          cur = cur.parentElement;
        }
        return node;
      };

      const parsePct = (txt) => {
        const t = String(txt || "");
        const m = t.match(/(\d+(?:\.\d+)?)\s*%/);
        return m ? m[1] : null;
      };

      const parsePrice = (txt) => {
        const t = String(txt || "");
        const m = t.match(/([+-]\d{2,5})/);
        return m ? m[1] : null;
      };

      const parseCellTwoRowsPrice = (cell) => {
        const out = [ { price: null }, { price: null } ];
        if (!cell) return out;

        const btns = Array.from(cell.querySelectorAll('span[role="button"]'));
        const texts = btns.map(b => norm(b.textContent)).filter(Boolean);

        for (let i = 0; i < Math.min(2, texts.length); i++) {
          out[i] = { price: parsePrice(texts[i]) };
        }
        return out;
      };

      const rows = [];

      for (const eidNode of leftEids) {
        const eidRaw = eidNode.getAttribute("data-horizontal-eid");
        const event_id = eidRaw ? Number(eidRaw) : null;

        const root = ascendToGameRoot(eidNode);

        const timeEl = root.querySelector('[data-vertical-sbid="time"] span');
        const start_time = timeEl ? norm(timeEl.textContent) : null;

        const matchupA = root.querySelector('a[href^="/scores/nba-basketball/matchup/"]');
        const matchup_url = matchupA ? matchupA.getAttribute("href") : null;

        const leftCol = root.querySelector("div.col-3");
        const teamSpans = leftCol ? Array.from(leftCol.querySelectorAll("span.fw-bolder")) : [];
        const teams = teamSpans.map(s => norm(s.textContent)).filter(Boolean).slice(0, 2);

        const scoreEls = leftCol ? Array.from(leftCol.querySelectorAll('div[class*="scores"] div')) : [];
        const scores = scoreEls.map(s => norm(s.textContent)).filter(Boolean).slice(0, 2);

        const consPctCol = root.querySelector('[data-vertical-sbid="-2"]');
        const pctSpans = consPctCol ? Array.from(consPctCol.querySelectorAll("span.opener")) : [];
        const cons_pcts = pctSpans.map(s => parsePct(norm(s.textContent))).slice(0, 2);

        const consOpenCol = root.querySelector('[data-vertical-sbid="-1"]');
        const openRowSpans = consOpenCol
          ? Array.from(consOpenCol.querySelectorAll('span[data-cy="odd-grid-opener-homepage"]'))
          : [];
        const openPrices = openRowSpans.map(s => parsePrice(norm(s.textContent))).slice(0, 2);

        const cellDivs = Array.from(root.querySelectorAll('div[class*="OddsCells_numbersContainer"]'));
        const n = books.length ? Math.min(books.length, cellDivs.length) : cellDivs.length;
        const cells = cellDivs.slice(0, n).map(c => parseCellTwoRowsPrice(c));

        for (let r = 0; r < 2; r++) {
          const team_name = teams[r] || null;
          const score = scores[r] || null;

          const row = {
            event_id,
            start_time,
            matchup_url,
            row_index: r,
            team_row: (r === 0 ? "away" : "home"),
            team_name,
            score,
            consensus_pct: cons_pcts[r] || null,
            consensus_opener_price: openPrices[r] || null,
          };

          for (let j = 0; j < n; j++) {
            const book = books[j] || `book_${j}`;
            const two = cells[j] || [{price:null},{price:null}];
            const v = two[r] || { price: null };
            row[`book__${book}__price`] = v.price;
          }

          rows.push(row);
        }
      }

      return { books, rows };
    }
    """

    payload = await page.evaluate(js)
    rows: list[dict[str, Any]] = (payload or {}).get("rows") or []
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df.insert(0, "date", page_date.isoformat())

    def to_float(series: pd.Series) -> pd.Series:
        return pd.to_numeric(
            series.astype("string").str.replace("−", "-", regex=False), errors="coerce"
        )

    df["consensus_pct"] = to_float(df.get("consensus_pct", pd.Series(dtype="string")))
    df["consensus_opener_price"] = to_float(
        df.get("consensus_opener_price", pd.Series(dtype="string"))
    )

    for c in df.columns:
        if c.endswith("__price") and c.startswith("book__"):
            df[c] = to_float(df[c])

    rename_map: dict[str, str] = {}
    for c in df.columns:
        if c.startswith("book__") and c.endswith("__price"):
            parts = c.split("__")
            if len(parts) == 3:
                book_name = parts[1]
                rename_map[c] = f"{_slugify_book(book_name)}_price"
    df = df.rename(columns=rename_map)
    return df


async def extract_sbr_spread_full_game_rows(
    page: Page, page_date: date
) -> pd.DataFrame:
    js = r"""
    () => {
      const norm = (s) => (s || "").replace(/\s+/g, " ").trim();

      // Line and price sit in sibling child spans, so textContent glues them
      // ("-10.5-110") and the whitespace-anchored number regex finds nothing.
      // Join the children with a space; fall back to textContent otherwise.
      const buttonText = (node) => {
        const parts = Array.from(node.querySelectorAll("span"))
          .map(s => norm(s.textContent))
          .filter(Boolean);
        return parts.length ? parts.join(" ") : norm(node.textContent);
      };

      const section = document.querySelector("#section-nba");
      if (!section) return { books: [], rows: [] };

      const thead = section.querySelector("#thead-nba");
      let books = [];
      if (thead) {
        const imgs = Array.from(thead.querySelectorAll("a img[alt]"));
        books = imgs
          .map(img => norm(img.getAttribute("alt")))
          .map(t => t.replace(/\s+Logo$/i, ""))
          .filter(Boolean);
      }

      const tbody = section.querySelector("#tbody-nba");
      if (!tbody) return { books, rows: [] };

      const leftEids = Array.from(tbody.querySelectorAll("[data-horizontal-eid]"));

      const ascendToGameRoot = (node) => {
        let cur = node;
        while (cur && cur !== section) {
          const hasOdds = cur.querySelector && cur.querySelector('a[data-aatracker^="Odds Table - Odds Cell CTA"]');
          if (hasOdds) return cur;
          cur = cur.parentElement;
        }
        return node;
      };

      const parsePct = (txt) => {
        const t = String(txt || "");
        const m = t.match(/(\d+(?:\.\d+)?)\s*%/);
        return m ? m[1] : null;
      };

      const parseLineAndPrice = (txt) => {
        const t = String(txt || "");
        const nums = Array.from(
          t.matchAll(/(?:^|\s)([+-]\d+(?:\.\d+)?)(?=$|\s)/g)
        ).map(m => m[1]);

        let priceIdx = -1;
        for (let i = nums.length - 1; i >= 0; i--) {
          const n = Number(nums[i]);
          if (/^[+-]\d{2,4}$/.test(nums[i]) && Math.abs(n) >= 90) {
            priceIdx = i;
            break;
          }
        }

        const price = priceIdx >= 0 ? nums[priceIdx] : null;
        const line = (
          nums.find((v, i) => i !== priceIdx && Math.abs(Number(v)) <= 60)
          || null
        );
        return { line, price };
      };

      const parseCellTwoRows = (cell) => {
        const out = [
          { line: null, price: null },
          { line: null, price: null }
        ];
        if (!cell) return out;

        const btns = Array.from(cell.querySelectorAll('span[role="button"]'));
        const texts = btns.map(buttonText).filter(Boolean);

        for (let i = 0; i < Math.min(2, texts.length); i++) {
          out[i] = parseLineAndPrice(texts[i]);
        }
        return out;
      };

      const rows = [];

      for (const eidNode of leftEids) {
        const eidRaw = eidNode.getAttribute("data-horizontal-eid");
        const event_id = eidRaw ? Number(eidRaw) : null;

        const root = ascendToGameRoot(eidNode);

        const timeEl = root.querySelector('[data-vertical-sbid="time"] span');
        const start_time = timeEl ? norm(timeEl.textContent) : null;

        const matchupA = root.querySelector('a[href^="/scores/nba-basketball/matchup/"]');
        const matchup_url = matchupA ? matchupA.getAttribute("href") : null;

        const leftCol = root.querySelector("div.col-3");
        const teamSpans = leftCol ? Array.from(leftCol.querySelectorAll("span.fw-bolder")) : [];
        const teams = teamSpans.map(s => norm(s.textContent)).filter(Boolean).slice(0, 2);

        const scoreEls = leftCol ? Array.from(leftCol.querySelectorAll('div[class*="scores"] div')) : [];
        const scores = scoreEls.map(s => norm(s.textContent)).filter(Boolean).slice(0, 2);

        const consPctCol = root.querySelector('[data-vertical-sbid="-2"]');
        const pctSpans = consPctCol ? Array.from(consPctCol.querySelectorAll("span.opener")) : [];
        const cons_pcts = pctSpans.map(s => parsePct(norm(s.textContent))).slice(0, 2);

        const consOpenCol = root.querySelector('[data-vertical-sbid="-1"]');
        const openRowSpans = consOpenCol
          ? Array.from(consOpenCol.querySelectorAll('span[data-cy="odd-grid-opener-homepage"]'))
          : [];
        const openRows = openRowSpans.map(s => parseLineAndPrice(buttonText(s))).slice(0, 2);

        const cellDivs = Array.from(root.querySelectorAll('div[class*="OddsCells_numbersContainer"]'));
        const n = books.length ? Math.min(books.length, cellDivs.length) : cellDivs.length;
        const cells = cellDivs.slice(0, n).map(c => parseCellTwoRows(c));

        for (let r = 0; r < 2; r++) {
          const team_name = teams[r] || null;
          const score = scores[r] || null;

          const opener = openRows[r] || { line: null, price: null };

          const row = {
            event_id,
            start_time,
            matchup_url,
            row_index: r,
            team_row: (r === 0 ? "away" : "home"),
            team_name,
            score,
            consensus_pct: cons_pcts[r] || null,
            consensus_opener_spread_line: opener.line,
            consensus_opener_spread_price: opener.price,
          };

          for (let j = 0; j < n; j++) {
            const book = books[j] || `book_${j}`;
            const two = cells[j] || [{line:null,price:null},{line:null,price:null}];
            const v = two[r] || { line: null, price: null };
            row[`book__${book}__spread_line`] = v.line;
            row[`book__${book}__spread_price`] = v.price;
          }

          rows.push(row);
        }
      }

      return { books, rows };
    }
    """

    payload = await page.evaluate(js)
    rows: list[dict[str, Any]] = (payload or {}).get("rows") or []
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df.insert(0, "date", page_date.isoformat())

    def to_float(series: pd.Series) -> pd.Series:
        return pd.to_numeric(
            series.astype("string").str.replace("−", "-", regex=False), errors="coerce"
        )

    df["consensus_pct"] = to_float(df.get("consensus_pct", pd.Series(dtype="string")))
    df["consensus_opener_spread_line"] = to_float(
        df.get("consensus_opener_spread_line", pd.Series(dtype="string"))
    )
    df["consensus_opener_spread_price"] = to_float(
        df.get("consensus_opener_spread_price", pd.Series(dtype="string"))
    )

    for c in df.columns:
        if c.startswith("book__") and (
            c.endswith("__spread_line") or c.endswith("__spread_price")
        ):
            df[c] = to_float(df[c])

    rename_map: dict[str, str] = {}
    for c in df.columns:
        if c.startswith("book__") and (
            c.endswith("__spread_line") or c.endswith("__spread_price")
        ):
            parts = c.split("__")
            if len(parts) == 3:
                book_name = parts[1]
                suffix = parts[2]
                rename_map[c] = f"{_slugify_book(book_name)}_{suffix}"
    df = df.rename(columns=rename_map)
    return df


async def scrape_day_totals(page: Page, d: date) -> TotalsScrapeResult:
    url = build_sbr_totals_url(d)

    await page.goto(url, wait_until="domcontentloaded", timeout=TIMEOUT_MS)
    await try_click_consent(page)

    if await has_no_odds_message(page):
        return TotalsScrapeResult(df=None, no_odds=True)

    try:
        await page.wait_for_selector(
            "#section-nba #tbody-nba [data-horizontal-eid]", timeout=TIMEOUT_MS
        )
    except PlaywrightTimeoutError:
        return TotalsScrapeResult(df=None, no_odds=False)

    df = await extract_sbr_totals_full_game_rows(page, d)
    if df.empty:
        return TotalsScrapeResult(df=None, no_odds=False)

    df.insert(1, "season", season_year_for_date(d))
    return TotalsScrapeResult(df=df, no_odds=False)


async def scrape_day_moneyline(page: Page, d: date) -> pd.DataFrame | None:
    url = build_sbr_moneyline_url(d)

    await page.goto(url, wait_until="domcontentloaded", timeout=TIMEOUT_MS)
    await try_click_consent(page)

    if await has_no_odds_message(page):
        return None

    try:
        await page.wait_for_selector(
            "#section-nba #tbody-nba [data-horizontal-eid]", timeout=TIMEOUT_MS
        )
    except PlaywrightTimeoutError:
        return None

    df = await extract_sbr_moneyline_full_game_rows(page, d)
    if df.empty:
        return None

    df.insert(1, "season", season_year_for_date(d))
    return df


async def scrape_day_spread(page: Page, d: date) -> pd.DataFrame | None:
    url = build_sbr_spread_url(d)

    await page.goto(url, wait_until="domcontentloaded", timeout=TIMEOUT_MS)
    await try_click_consent(page)

    if await has_no_odds_message(page):
        return None

    try:
        await page.wait_for_selector(
            "#section-nba #tbody-nba [data-horizontal-eid]", timeout=TIMEOUT_MS
        )
    except PlaywrightTimeoutError:
        return None

    df = await extract_sbr_spread_full_game_rows(page, d)
    if df.empty:
        return None

    df.insert(1, "season", season_year_for_date(d))
    return df


# ---------------------------------------------------------------------------
# JSON engine
#
# Every odds page is server-rendered by Next.js and ships the table it draws as
# JSON in <script id="__NEXT_DATA__">. One plain GET (~0.2 s) replaces a browser
# page load (~6 s), with no cookie banner and no dependence on CSS class names.
#
# The parsers below return exactly the raw frames the DOM extractors above
# return, so everything downstream is unchanged. Where the DOM extractors run a
# regex over the *rendered* text, the value is rendered the same way and the
# same regex applied, rather than read straight from the JSON — so even their
# quirks (a 2147483647 sentinel price truncated to 21474, a spread price under
# 90 dropped) come out identically.
#
# The one intended difference: a game with no odds at all has no odds cells,
# so the DOM extractor ascends past its own row and copies the neighbouring
# game's teams and odds under its event_id. Here it gets its own teams and
# empty odds instead. Those ghost rows never matched a scheduled game.
# ---------------------------------------------------------------------------

NEXT_DATA_RE = re.compile(r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>', re.DOTALL)
REQUEST_HEADERS = {"User-Agent": "Mozilla/5.0"}
REQUEST_TIMEOUT_S = 30
REQUEST_RETRIES = 3
JSON_SLEEP_MIN_S = 0.2
JSON_SLEEP_MAX_S = 0.5
JSON_MAX_WORKERS = 4
# The table header renders the first seven sportsbooks of the payload's list;
# the eighth (Hard Rock Bet) is carried in the JSON but never drawn.
RENDERED_SPORTSBOOKS = 7


def fetch_sbr_page(session: requests.Session, url: str) -> str:
    """GET an SBR page, retrying transient failures with a short backoff."""
    for attempt in range(1, REQUEST_RETRIES + 1):
        try:
            response = session.get(
                url, headers=REQUEST_HEADERS, timeout=REQUEST_TIMEOUT_S
            )
            if response.status_code == 200:
                return response.text
            error: Exception = RuntimeError(f"HTTP {response.status_code} for {url}")
        except requests.RequestException as exc:
            error = exc
        if attempt < REQUEST_RETRIES:
            time.sleep(2**attempt)
    raise error


def parse_sbr_odds_table(html: str) -> dict[str, Any] | None:
    """The NBA odds table model from a page, or None when it lists no games.

    A page without ``__NEXT_DATA__`` raises: that is SBR changing the site, and
    must not pass silently as "no games that day".
    """
    match = NEXT_DATA_RE.search(html)
    if not match:
        raise ValueError("SBR page has no __NEXT_DATA__ payload")
    page_props = json.loads(match.group(1)).get("props", {}).get("pageProps", {})
    for table in page_props.get("oddsTables") or []:
        if str(table.get("league") or "").upper() == "NBA":
            model = table.get("oddsTableModel") or {}
            return model if model.get("gameRows") else None
    return None


def _rendered_price(price: Any, max_digits: int) -> str | None:
    if price is None:
        return None
    match = re.search(rf"([+-]\d{{2,{max_digits}}})", f"{int(price):+d}")
    return match.group(1) if match else None


def _totals_line_and_price(line: Any, price: Any) -> tuple[str | None, str | None]:
    """parseLineAndPrice (totals) applied to the rendered cell text."""
    parts = []
    if line is not None:
        parts.append(f"{float(line):g}")
    if price is not None:
        parts.append(f"{int(price):+d}")
    text = " ".join(parts)
    line_match = re.search(r"([0-9]+(?:\.[0-9]+)?)", text)
    price_match = re.search(r"([+-]\d{2,4})", text)
    return (
        line_match.group(1) if line_match else None,
        price_match.group(1) if price_match else None,
    )


def _spread_line_and_price(line: Any, price: Any) -> tuple[str | None, str | None]:
    """parseLineAndPrice (spread) applied to the rendered cell text."""
    parts = []
    if line is not None:
        # A pick'em renders "PK", which the signed-number regex never matches.
        parts.append("PK" if float(line) == 0 else f"{float(line):+g}")
    if price is not None:
        parts.append(f"{int(price):+d}")
    nums = re.findall(r"(?:^|\s)([+-]\d+(?:\.\d+)?)(?=$|\s)", " ".join(parts))
    price_idx = next(
        (
            i
            for i in range(len(nums) - 1, -1, -1)
            if re.fullmatch(r"[+-]\d{2,4}", nums[i]) and abs(float(nums[i])) >= 90
        ),
        -1,
    )
    spread_price = nums[price_idx] if price_idx >= 0 else None
    spread_line = next(
        (v for i, v in enumerate(nums) if i != price_idx and abs(float(v)) <= 60),
        None,
    )
    return spread_line, spread_price


def _rendered_pct(value: Any) -> str | None:
    # The consensus column shows a rounded integer and nothing for 0.
    if not value:
        return None
    return str(int(float(value) + 0.5))


def _raw_rows(model: dict[str, Any], market: str, fill_row) -> list[dict[str, Any]]:
    books = [
        _slugify_book(b.get("name") or f"book_{j}")
        for j, b in enumerate(model.get("sportsbooks") or [])
    ][:RENDERED_SPORTSBOOKS]
    rows = []
    for game in model["gameRows"]:
        view = game["gameView"]
        event_id = view["gameId"]
        odds_views = game.get("oddsViews") or []
        opening_views = game.get("openingLineViews") or []
        opening = ((opening_views[0] if opening_views else None) or {}).get(
            "openingLine"
        ) or {}
        current = [
            ((odds_views[j] if j < len(odds_views) else None) or {}).get("currentLine")
            or {}
            for j in range(len(books))
        ]
        consensus = view.get("consensus") or {}
        for row_index, side in enumerate(("away", "home")):
            score = view.get(f"{side}TeamScore")
            row: dict[str, Any] = {
                "event_id": event_id,
                "start_time": view.get("startDate"),
                "matchup_url": f"/scores/nba-basketball/matchup/{event_id}/",
                "row_index": row_index,
            }
            if market != "totals":
                row["team_row"] = side
            row["team_name"] = (view.get(f"{side}Team") or {}).get("displayName")
            row["score"] = None if score is None else str(score)
            fill_row(row, row_index, side, books, current, opening, consensus)
            rows.append(row)
    return rows


def _raw_frame(rows: list[dict[str, Any]], page_date: date) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df.insert(0, "date", page_date.isoformat())
    numeric = [
        c
        for c in df.columns
        if c in ("consensus_pct",)
        or (c.startswith("consensus_opener_") and c != "consensus_opener_side")
        or c.endswith(("_line", "_price"))
    ]
    for c in numeric:
        df[c] = pd.to_numeric(df[c].astype("string"), errors="coerce")
    return df


def parse_sbr_totals_rows(model: dict[str, Any], page_date: date) -> pd.DataFrame:
    def fill(row, row_index, side, books, current, opening, consensus):
        key = "over" if row_index == 0 else "under"
        row["consensus_pct"] = _rendered_pct(consensus.get(f"{key}PickPercent"))
        row["consensus_opener_side"] = "O" if row_index == 0 else "U"
        row["consensus_opener_line"], row["consensus_opener_price"] = (
            _totals_line_and_price(opening.get("total"), opening.get(f"{key}Odds"))
        )
        for book, line in zip(books, current, strict=True):
            row[f"{book}_line"], row[f"{book}_price"] = _totals_line_and_price(
                line.get("total"), line.get(f"{key}Odds")
            )

    return _raw_frame(_raw_rows(model, "totals", fill), page_date)


def parse_sbr_spread_rows(model: dict[str, Any], page_date: date) -> pd.DataFrame:
    def fill(row, row_index, side, books, current, opening, consensus):
        row["consensus_pct"] = _rendered_pct(consensus.get(f"{side}SpreadPickPercent"))
        (
            row["consensus_opener_spread_line"],
            row["consensus_opener_spread_price"],
        ) = _spread_line_and_price(
            opening.get(f"{side}Spread"), opening.get(f"{side}Odds")
        )
        for book, line in zip(books, current, strict=True):
            row[f"{book}_spread_line"], row[f"{book}_spread_price"] = (
                _spread_line_and_price(
                    line.get(f"{side}Spread"), line.get(f"{side}Odds")
                )
            )

    return _raw_frame(_raw_rows(model, "spread", fill), page_date)


def parse_sbr_moneyline_rows(model: dict[str, Any], page_date: date) -> pd.DataFrame:
    def fill(row, row_index, side, books, current, opening, consensus):
        row["consensus_pct"] = _rendered_pct(
            consensus.get(f"{side}MoneyLinePickPercent")
        )
        row["consensus_opener_price"] = _rendered_price(opening.get(f"{side}Odds"), 5)
        for book, line in zip(books, current, strict=True):
            row[f"{book}_price"] = _rendered_price(line.get(f"{side}Odds"), 5)

    return _raw_frame(_raw_rows(model, "ml", fill), page_date)


def _with_season(df: pd.DataFrame, d: date) -> pd.DataFrame | None:
    if df.empty:
        return None
    df.insert(1, "season", season_year_for_date(d))
    return df


def _merge_day(
    totals_raw: pd.DataFrame,
    spread_raw: pd.DataFrame | None,
    ml_raw: pd.DataFrame | None,
) -> pd.DataFrame:
    totals_game_df = load_one_day_totals_csv(totals_raw)
    spread_game_df = (
        load_one_day_spread_csv(spread_raw)
        if spread_raw is not None
        else pd.DataFrame(columns=["game_id"])
    )
    ml_game_df = (
        load_one_day_moneyline_csv(ml_raw)
        if ml_raw is not None
        else pd.DataFrame(columns=["game_id"])
    )
    return merge_daily_frames(totals_game_df, spread_game_df, ml_game_df)


def scrape_day_json(session: requests.Session, d: date) -> pd.DataFrame | None:
    """One date through the JSON engine, merged; None when it has no odds."""
    totals_model = parse_sbr_odds_table(
        fetch_sbr_page(session, build_sbr_totals_url(d))
    )
    if totals_model is None:
        return None
    totals_raw = _with_season(parse_sbr_totals_rows(totals_model, d), d)
    if totals_raw is None:
        return None

    raw: dict[str, pd.DataFrame | None] = {}
    for name, url, parse in (
        ("spread", build_sbr_spread_url(d), parse_sbr_spread_rows),
        ("ml", build_sbr_moneyline_url(d), parse_sbr_moneyline_rows),
    ):
        time.sleep(random.uniform(JSON_SLEEP_MIN_S, JSON_SLEEP_MAX_S))
        model = parse_sbr_odds_table(fetch_sbr_page(session, url))
        raw[name] = None if model is None else _with_season(parse(model, d), d)

    return _merge_day(totals_raw, raw["spread"], raw["ml"])


def scrape_sportsbook_days_json(
    days: list[date], *, max_workers: int = JSON_MAX_WORKERS
) -> pd.DataFrame:
    """JSON engine over ``days``; a few dates in flight at once.

    SBR renders each page on request (~1 s uncached), so the wait is on its
    side and a small pool divides the wall time. Output keeps ``days`` order.
    """
    local = threading.local()

    def one_day(d: date) -> pd.DataFrame | None:
        if not hasattr(local, "session"):
            local.session = requests.Session()
        try:
            return scrape_day_json(local.session, d)
        except Exception as e:
            print(f"Failed sportsbook scrape for {d.isoformat()}: {e}")
            return None
        finally:
            time.sleep(random.uniform(JSON_SLEEP_MIN_S, JSON_SLEEP_MAX_S))

    with ThreadPoolExecutor(max_workers=max(1, max_workers)) as pool:
        results = list(
            tqdm(
                pool.map(one_day, days),
                total=len(days),
                desc="Scraping sportsbook days",
                unit="day",
            )
        )

    merged_days = [df for df in results if df is not None]
    if not merged_days:
        return pd.DataFrame()
    return pd.concat(merged_days, ignore_index=True)


async def scrape_sportsbook_days(
    days: list[date],
    *,
    headless: bool = True,
    engine: str = "json",
) -> pd.DataFrame:
    """Closing odds for ``days``, one merged row per game.

    ``engine="json"`` (default) reads the embedded payload over plain HTTP;
    ``engine="browser"`` drives Playwright through the rendered table and is
    kept as a fallback. ``headless`` only applies to the browser engine.
    """
    if engine == "json":
        return await asyncio.to_thread(scrape_sportsbook_days_json, list(days))
    if engine == "browser":
        return await scrape_sportsbook_days_browser(days, headless=headless)
    raise ValueError(f"Unknown sportsbook scrape engine: {engine!r}")


async def scrape_sportsbook_days_browser(
    days: list[date],
    *,
    headless: bool = True,
) -> pd.DataFrame:
    if not days:
        return pd.DataFrame()

    merged_days: list[pd.DataFrame] = []

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=headless)
        context = await browser.new_context()
        page = await context.new_page()

        for d in tqdm(days, desc="Scraping sportsbook days", unit="day"):
            try:
                totals_result = await scrape_day_totals(page, d)
                if totals_result.no_odds or totals_result.df is None:
                    await random_sleep()
                    continue

                spread_df_raw = await scrape_day_spread(page, d)
                ml_df_raw = await scrape_day_moneyline(page, d)
                merged_days.append(
                    _merge_day(totals_result.df, spread_df_raw, ml_df_raw)
                )

            except Exception as e:
                print(f"Failed sportsbook scrape for {d.isoformat()}: {e}")

            await random_sleep()

        await browser.close()

    if not merged_days:
        return pd.DataFrame()

    return pd.concat(merged_days, ignore_index=True)
