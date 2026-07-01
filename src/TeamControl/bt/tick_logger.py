"""Per-tick BT logger.

Writes one entry per tick to out/<session>/ticks.log (human-readable) and
out/<session>/ticks.jsonl (machine-readable JSONL).

Each entry captures:
  - tick number and wall-clock timestamp
  - game phase
  - per robot: role, intent type, intent target coordinates, and the BT node
    that produced the intent (intent_source)

Usage::

    logger = TickLogger(robot_ids=[0, 1, 2, 3, 4, 5], team="yellow")
    # inside the tick loop, after coordinator.tick():
    logger.log_tick(tick_count, snapshot, coordinator.blackboards, robot_ids)
    # on shutdown:
    logger.close()
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from TeamControl.bt.contracts.blackboard import RobotBlackboard
    from TeamControl.bt.contracts.snapshot import Snapshot


def _format_intent(intent) -> tuple[str, str]:
    """Return (intent_type, intent_detail) strings for one intent."""
    if intent is None:
        return "None", ""
    kind = type(intent).__name__.replace("Intent", "")
    parts: list[str] = []
    if hasattr(intent, "target_pos") and intent.target_pos is not None:
        x, y = intent.target_pos
        parts.append(f"pos=({x:.3f},{y:.3f})")
    if hasattr(intent, "target_robot_id"):
        parts.append(f"to=R{intent.target_robot_id}")
    if hasattr(intent, "target_orientation") and intent.target_orientation is not None:
        parts.append(f"ori={intent.target_orientation:.3f}rad")
    return kind, " ".join(parts)


def _intent_dict(intent) -> dict:
    """Serialize one intent to a plain dict for JSONL output."""
    if intent is None:
        return {"type": None}
    kind = type(intent).__name__.replace("Intent", "")
    d: dict = {"type": kind}
    if hasattr(intent, "target_pos") and intent.target_pos is not None:
        d["target_pos"] = list(intent.target_pos)
    if hasattr(intent, "target_robot_id"):
        d["target_robot_id"] = intent.target_robot_id
    if hasattr(intent, "target_orientation") and intent.target_orientation is not None:
        d["target_orientation"] = intent.target_orientation
    return d


class TickLogger:
    """Writes per-tick BT state to out/<session>/.

    Parameters
    ----------
    robot_ids:
        Ordered list of robot IDs that will appear in every log entry.
    team:
        Short label for the team ("yellow" / "blue") used in the session
        directory name and log headers.
    out_root:
        Root output directory.  Defaults to ``out/`` relative to CWD (i.e.
        the project root when launched normally).
    """

    def __init__(
        self,
        robot_ids: list[int],
        team: str = "unknown",
        out_root: str | Path = "out",
    ) -> None:
        self._robot_ids = robot_ids
        session_ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        session_dir = Path(out_root) / f"{session_ts}_{team}"
        session_dir.mkdir(parents=True, exist_ok=True)

        self._log_path = session_dir / "ticks.log"
        self._jsonl_path = session_dir / "ticks.jsonl"

        self._log_file = open(self._log_path, "w", buffering=1, encoding="utf-8")
        self._jsonl_file = open(self._jsonl_path, "w", buffering=1, encoding="utf-8")

        header = (
            f"# BT Tick Log — team={team}  session={session_ts}\n"
            f"# robots={robot_ids}\n"
            f"# columns: tick | timestamp | phase | robot_id | role | intent | detail | source\n"
            f"{'=' * 100}\n"
        )
        self._log_file.write(header)

    # ------------------------------------------------------------------
    def log_tick(
        self,
        tick_count: int,
        snapshot: "Snapshot",
        blackboards: dict[int, "RobotBlackboard"],
        robot_ids: list[int] | None = None,
    ) -> None:
        """Write one tick entry for all robots.

        Parameters
        ----------
        tick_count:
            Current tick counter (monotonically increasing).
        snapshot:
            Immutable world-state snapshot for this tick.
        blackboards:
            ``coordinator.blackboards`` dict mapping robot_id → RobotBlackboard.
        robot_ids:
            Subset of robot IDs to log.  Defaults to the list provided at
            construction time.
        """
        ids = robot_ids if robot_ids is not None else self._robot_ids
        ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        phase = snapshot.referee_state.game_phase.value

        # --- human-readable ------------------------------------------------
        self._log_file.write(f"\n[tick={tick_count:06d}  {ts}  phase={phase}]\n")

        robots_json: list[dict] = []
        for rid in ids:
            bb = blackboards.get(rid)
            if bb is None:
                self._log_file.write(f"  R{rid}: <no blackboard>\n")
                robots_json.append({"id": rid, "role": None, "intent": {"type": None}, "source": None})
                continue

            role = bb.current_role.value
            intent_type, intent_detail = _format_intent(bb.current_intent)
            source = bb.intent_source or "—"

            self._log_file.write(
                f"  R{rid}  role={role:<12s}  intent={intent_type:<10s}  "
                f"{intent_detail:<35s}  source={source}\n"
            )
            robots_json.append(
                {
                    "id": rid,
                    "role": role,
                    "intent": _intent_dict(bb.current_intent),
                    "source": bb.intent_source,
                }
            )

        # --- JSONL ---------------------------------------------------------
        record = {
            "tick": tick_count,
            "ts": ts,
            "phase": phase,
            "robots": robots_json,
        }
        self._jsonl_file.write(json.dumps(record) + "\n")

    # ------------------------------------------------------------------
    def close(self) -> None:
        """Flush and close both output files."""
        for f in (self._log_file, self._jsonl_file):
            try:
                f.flush()
                f.close()
            except Exception:
                pass
