"""Tests for cross-process sharing of the Bilibili search backoff state."""

from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING

import pytest

from openbiliclaw.bilibili import search_backoff
from openbiliclaw.bilibili.api import BilibiliAPIClient
from openbiliclaw.bilibili.search_backoff import configure_search_backoff_state_path

if TYPE_CHECKING:
    from pathlib import Path

_STATE_FILENAME = "bilibili_search_backoff.json"


@pytest.fixture(autouse=True)
def _reset_bilibili_search_cooldown(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(BilibiliAPIClient, "_search_cooldown_until", 0.0)
    monkeypatch.setattr(BilibiliAPIClient, "_search_cooldown_level", 0)
    monkeypatch.setattr(BilibiliAPIClient, "_search_voucher_block_streak", 0)
    monkeypatch.setattr(BilibiliAPIClient, "_search_dom_fallback_until", 0.0)


def _simulate_fresh_process(monkeypatch: pytest.MonkeyPatch) -> None:
    """Zero the in-process ClassVars, as a newly spawned process would see them."""
    monkeypatch.setattr(BilibiliAPIClient, "_search_cooldown_until", 0.0)
    monkeypatch.setattr(BilibiliAPIClient, "_search_cooldown_level", 0)
    monkeypatch.setattr(BilibiliAPIClient, "_search_voucher_block_streak", 0)
    monkeypatch.setattr(BilibiliAPIClient, "_search_dom_fallback_until", 0.0)


def _state_file(tmp_path: Path) -> Path:
    return tmp_path / _STATE_FILENAME


def test_412_cooldown_shared_across_processes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    BilibiliAPIClient._activate_search_cooldown(
        base_seconds=BilibiliAPIClient._SEARCH_COOLDOWN_412_SECONDS
    )

    _simulate_fresh_process(monkeypatch)

    remaining = BilibiliAPIClient.search_cooldown_remaining()
    assert 590.0 < remaining <= 600.0


def test_dom_fallback_shared_across_processes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    BilibiliAPIClient._activate_search_dom_fallback()

    _simulate_fresh_process(monkeypatch)

    remaining = BilibiliAPIClient.search_dom_fallback_remaining()
    assert 170.0 < remaining <= 180.0


def test_voucher_streak_and_level_shared_across_processes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    BilibiliAPIClient._record_voucher_block()
    BilibiliAPIClient._record_voucher_block()

    _simulate_fresh_process(monkeypatch)
    BilibiliAPIClient.search_cooldown_remaining()  # pulls the shared state in

    assert BilibiliAPIClient._search_voucher_block_streak == 2


def test_most_conservative_deadline_wins(tmp_path: Path) -> None:
    BilibiliAPIClient._activate_search_cooldown(base_seconds=600.0)
    # A second process that only knows a shorter cooldown must not shorten
    # the persisted one.
    shorter = search_backoff.snapshot_from_monotonic(
        cooldown_until=time.monotonic() + 60.0,
        cooldown_level=1,
        voucher_block_streak=0,
        dom_fallback_until=0.0,
    )
    search_backoff.persist_shared_backoff(shorter)

    shared = search_backoff.read_shared_backoff()
    assert shared is not None
    assert shared.cooldown_until - time.time() > 590.0


def test_reset_counters_propagates_to_disk(tmp_path: Path) -> None:
    BilibiliAPIClient._record_voucher_block()
    BilibiliAPIClient._record_voucher_block()

    BilibiliAPIClient._reset_search_cooldown_backoff()

    shared = search_backoff.read_shared_backoff()
    assert shared is not None
    assert shared.voucher_block_streak == 0
    assert shared.cooldown_level == 0


def test_stale_counters_do_not_trip_fresh_process(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    stale = {
        "version": 1,
        "scope": "global",
        "cooldown_until": 0.0,
        "cooldown_level": 3,
        "voucher_block_streak": 5,
        "dom_fallback_until": 0.0,
        "updated_at": time.time() - 7200.0,
    }
    _state_file(tmp_path).write_text(json.dumps(stale), encoding="utf-8")

    _simulate_fresh_process(monkeypatch)

    duration = BilibiliAPIClient._record_voucher_block()
    assert duration == 0.0
    assert BilibiliAPIClient._search_voucher_block_streak == 1
    assert BilibiliAPIClient._search_cooldown_level == 0


def test_persistence_disabled_matches_inprocess_behavior(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configure_search_backoff_state_path(None)

    BilibiliAPIClient._activate_search_cooldown(base_seconds=600.0)
    assert BilibiliAPIClient.search_cooldown_remaining() > 590.0

    _simulate_fresh_process(monkeypatch)
    assert BilibiliAPIClient.search_cooldown_remaining() == 0.0


def test_unwritable_state_path_fails_open(tmp_path: Path) -> None:
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory", encoding="utf-8")
    configure_search_backoff_state_path(blocker / "state.json")

    duration = BilibiliAPIClient._activate_search_cooldown(base_seconds=600.0)

    assert duration == 600.0
    assert BilibiliAPIClient.search_cooldown_remaining() > 590.0


def test_state_file_schema(tmp_path: Path) -> None:
    BilibiliAPIClient._activate_search_cooldown(base_seconds=600.0)

    raw = json.loads(_state_file(tmp_path).read_text(encoding="utf-8"))
    assert raw["version"] == 1
    assert raw["scope"] == "global"
    assert raw["cooldown_until"] > time.time() + 590.0
    assert raw["cooldown_level"] == 1
    assert raw["dom_fallback_until"] > time.time() + 590.0
    assert raw["updated_at"] > 0.0
