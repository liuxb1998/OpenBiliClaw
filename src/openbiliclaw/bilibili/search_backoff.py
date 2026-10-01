"""Cross-process persistence for the Bilibili search cooldown/backoff state.

The API client keeps its search cooldown in process-global ClassVars, but the
CLI's multi-process layout (API server + worker + discovery worker) gives each
process its own copy: one process can sit inside a 412 hard cooldown while
another keeps hammering the same exit IP. This module mirrors the four
counters to a small JSON state file under the data directory so every process
backs off — and recovers — together.

Persistence is strictly best-effort: any I/O failure falls back to the
previous in-process behavior (fail-open, no config item, no switch). All
deadlines are stored as wall-clock epochs because ``time.monotonic()`` is not
comparable across processes.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

from openbiliclaw.memory.json_state import read_json_state, update_json_state

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)

_STATE_FILENAME = "bilibili_search_backoff.json"
_STATE_VERSION = 1
# Persisted counters (escalation level / v_voucher streak) describe one
# unfolding incident; ignore them once older than the longest possible
# cooldown so a stale half-finished incident cannot trip a fresh process on
# its first transient block.
_COUNTER_TTL_SECONDS = 1800.0

_UNSET: Any = object()
_state_path_override: Any = _UNSET
_default_path_resolved = False
_resolved_default_path: Path | None = None


@dataclass(frozen=True)
class SharedBackoff:
    """The persisted backoff state; deadlines are wall-clock epochs."""

    cooldown_until: float = 0.0
    cooldown_level: int = 0
    voucher_block_streak: int = 0
    dom_fallback_until: float = 0.0
    updated_at: float = 0.0

    def counters_fresh(self, now: float) -> bool:
        """Whether the persisted counters still belong to a live incident."""
        return 0.0 <= now - self.updated_at <= _COUNTER_TTL_SECONDS


def configure_search_backoff_state_path(path: Path | None) -> None:
    """Force the shared-state location; ``None`` disables persistence.

    Production code never calls this — the default resolves to
    ``<data_dir>/bilibili_search_backoff.json``. Tests use it to isolate the
    on-disk state (tmp path) or to verify the fail-open behavior (``None``).
    """
    global _state_path_override
    _state_path_override = path


def _state_path() -> Path | None:
    global _default_path_resolved, _resolved_default_path
    if _state_path_override is not _UNSET:
        path: Path | None = _state_path_override
        return path
    if not _default_path_resolved:
        _default_path_resolved = True
        try:
            from openbiliclaw.config import load_config

            _resolved_default_path = load_config().data_path / _STATE_FILENAME
        except Exception:
            logger.debug(
                "bilibili search backoff: cannot resolve data dir — persistence disabled",
                exc_info=True,
            )
            _resolved_default_path = None
    return _resolved_default_path


def _as_float(value: object) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0


def _normalize(raw: Any) -> SharedBackoff:
    if not isinstance(raw, dict):
        return SharedBackoff()
    return SharedBackoff(
        cooldown_until=max(0.0, _as_float(raw.get("cooldown_until"))),
        cooldown_level=max(0, int(_as_float(raw.get("cooldown_level")))),
        voucher_block_streak=max(0, int(_as_float(raw.get("voucher_block_streak")))),
        dom_fallback_until=max(0.0, _as_float(raw.get("dom_fallback_until"))),
        updated_at=max(0.0, _as_float(raw.get("updated_at"))),
    )


def _serialize(state: SharedBackoff) -> dict[str, object]:
    return {
        "version": _STATE_VERSION,
        # Reserved: today the backoff is process-global, matching the exit-IP
        # scope of a 412 block; per-cookie/proxy scopes may split it later.
        "scope": "global",
        "cooldown_until": state.cooldown_until,
        "cooldown_level": state.cooldown_level,
        "voucher_block_streak": state.voucher_block_streak,
        "dom_fallback_until": state.dom_fallback_until,
        "updated_at": state.updated_at,
    }


def read_shared_backoff() -> SharedBackoff | None:
    """Read the persisted state; ``None`` when persistence is off/unavailable."""
    path = _state_path()
    if path is None:
        return None
    try:
        return read_json_state(
            path,
            default_factory=SharedBackoff,
            normalize=_normalize,
        )
    except Exception:
        logger.debug("bilibili search backoff: read failed — in-process state only", exc_info=True)
        return None


def persist_shared_backoff(
    snapshot: SharedBackoff,
    *,
    reset_counters: bool = False,
) -> None:
    """Merge ``snapshot`` into the persisted state (read-modify-write).

    Deadlines merge with ``max`` so the most conservative process wins. The
    escalation level and v_voucher streak also merge with ``max`` — except
    when ``reset_counters`` is set (a search succeeded), in which case the
    snapshot's zeroed counters overwrite so the recovery propagates.
    """

    def mutate(shared: SharedBackoff) -> SharedBackoff:
        if reset_counters:
            level = snapshot.cooldown_level
            streak = snapshot.voucher_block_streak
        elif shared.counters_fresh(snapshot.updated_at):
            level = max(shared.cooldown_level, snapshot.cooldown_level)
            streak = max(shared.voucher_block_streak, snapshot.voucher_block_streak)
        else:
            level = snapshot.cooldown_level
            streak = snapshot.voucher_block_streak
        return replace(
            shared,
            cooldown_until=max(shared.cooldown_until, snapshot.cooldown_until),
            cooldown_level=level,
            voucher_block_streak=streak,
            dom_fallback_until=max(shared.dom_fallback_until, snapshot.dom_fallback_until),
            updated_at=snapshot.updated_at,
        )

    path = _state_path()
    if path is None:
        return
    try:
        update_json_state(
            path,
            default_factory=SharedBackoff,
            normalize=_normalize,
            serialize=_serialize,
            mutate=mutate,
        )
    except Exception:
        logger.debug(
            "bilibili search backoff: persist failed — in-process state only", exc_info=True
        )


def _monotonic_to_wall(deadline: float, *, now_wall: float, now_monotonic: float) -> float:
    if deadline <= 0.0:
        return 0.0  # unset stays unset instead of becoming "expires right now"
    return now_wall + max(0.0, deadline - now_monotonic)


def snapshot_from_monotonic(
    *,
    cooldown_until: float,
    cooldown_level: int,
    voucher_block_streak: int,
    dom_fallback_until: float,
) -> SharedBackoff:
    """Convert the client's monotonic-clock ClassVars into a persistable snapshot."""
    now_wall = time.time()
    now_monotonic = time.monotonic()
    return SharedBackoff(
        cooldown_until=_monotonic_to_wall(
            cooldown_until, now_wall=now_wall, now_monotonic=now_monotonic
        ),
        cooldown_level=cooldown_level,
        voucher_block_streak=voucher_block_streak,
        dom_fallback_until=_monotonic_to_wall(
            dom_fallback_until, now_wall=now_wall, now_monotonic=now_monotonic
        ),
        updated_at=now_wall,
    )
