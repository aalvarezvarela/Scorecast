"""Jobs for ``nba_ou.prediction.side_build`` tests: a spawned side process
imports them by name, so they live in a module, not in a test."""

from __future__ import annotations

import os
import time

from nba_ou.utils import warm_gate


def add(a, b, *, c=0):
    print("adding")
    return a + b + c


def slow(seconds):
    """Prints, stays quiet ``seconds``, prints again; the last line has no
    newline."""
    print("starting", flush=True)
    time.sleep(seconds)
    print("finishing", end="", flush=True)
    return "done"


def fail():
    raise ValueError("boom")


def die():
    os._exit(9)


def cold_past_season(season):
    """A memo miss of ``season``; returns when the gate lets it go on."""
    warm_gate.cold("player_history_rows", season)
    return time.time()
