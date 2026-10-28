"""Loads of stored history shared in the process: one read per set of
arguments inside the scope, a copy for every caller, none outside."""

from __future__ import annotations

import threading

import numpy as np
import pandas as pd
from nba_ou.utils import history_cache
from nba_ou.utils.history_cache import cached


def _counting_load(**decorator):
    calls = []

    @cached(**decorator)
    def load(season_years=None, conn=None, verbose=True):
        calls.append(season_years)
        return pd.DataFrame({"season": list(season_years or [0])})

    return load, calls


def test_outside_the_scope_every_call_reads():
    load, calls = _counting_load()
    load([2024])
    load([2024])
    assert len(calls) == 2
    assert not history_cache.active()


def test_inside_the_scope_one_read_per_arguments():
    load, calls = _counting_load()
    with history_cache.history_cache():
        load([2024, 2025])
        load(season_years=[2024, 2025])
        load(np.array([2024, 2025]))
        load([2025])
    assert calls == [[2024, 2025], [2025]]


def test_ignored_arguments_share_the_read():
    load, calls = _counting_load(ignore=("conn", "verbose"))
    with history_cache.history_cache():
        load([2024], conn=object(), verbose=True)
        load([2024], conn=object(), verbose=False)
    assert len(calls) == 1


def test_an_argument_without_a_key_reads_every_time():
    load, calls = _counting_load()
    with history_cache.history_cache():
        load([2024], conn=object())
        load([2024], conn=object())
    assert len(calls) == 2


def test_each_caller_gets_its_own_copy():
    @cached()
    def load():
        return pd.DataFrame({"x": [1, 2]}), {"fingerprints": {2025: "a"}}

    with history_cache.history_cache():
        frame, extra = load()
        frame.loc[0, "x"] = 99
        extra["fingerprints"][2025] = "changed"
        again, extra_again = load()
    assert again["x"].tolist() == [1, 2]
    assert extra_again == {"fingerprints": {2025: "a"}}


def test_clear_and_leaving_the_scope_forget_the_reads():
    load, calls = _counting_load()
    with history_cache.history_cache():
        load([2024])
        history_cache.clear()
        load([2024])
        with history_cache.history_cache():
            load([2024])
    with history_cache.history_cache():
        load([2024])
    assert len(calls) == 3
    assert not history_cache.active()


def test_threads_asking_for_one_load_wait_for_a_single_read():
    started, release = threading.Event(), threading.Event()
    calls = []

    @cached()
    def load(season):
        calls.append(season)
        started.set()
        release.wait(5)
        return pd.DataFrame({"season": [season]})

    results = []
    with history_cache.history_cache():
        threads = [
            threading.Thread(target=lambda: results.append(load(2025)))
            for _ in range(3)
        ]
        for thread in threads:
            thread.start()
        started.wait(5)
        release.set()
        for thread in threads:
            thread.join(5)
    assert calls == [2025]
    assert [frame["season"].tolist() for frame in results] == [[2025]] * 3
