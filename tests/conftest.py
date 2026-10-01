"""Shared pytest fixtures for the OpenBiliClaw test suite."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from openbiliclaw.bilibili import search_backoff

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture(autouse=True)
def _isolate_bilibili_search_backoff(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Redirect the shared search-backoff state file into a tmp dir.

    The Bilibili API client persists its search cooldown so every runtime
    process backs off together; without this isolation the suite would read
    and write the real ``data/bilibili_search_backoff.json``.
    """
    monkeypatch.setattr(
        search_backoff,
        "_state_path_override",
        tmp_path / "bilibili_search_backoff.json",
    )
