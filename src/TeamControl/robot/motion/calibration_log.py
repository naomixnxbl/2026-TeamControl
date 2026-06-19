"""
Per-robot calibration history.

Every PD auto-tune run (`PDCalibration.auto_tune_turn` / `auto_tune_linear`
in pd_calibration.py) writes one timestamped JSON file here recording every
candidate gain pair it tried, not just the final winner — so a sweep's full
history can be inspected or replayed later.

`movement_calibration.json` (PDSettingsStore) only ever keeps the latest
winning gains; this module is the append-only history sitting next to it.

Layout:
    calibration_logs/<team>/<letter>/<timestamp>_<kind>_autotune.json

Hardware Auto-Calibrate (`ui/calibration_page.py`'s Auto-Calibrate / Speed
Sweep) does not use this yet — it still logs into the flat "runs" array
inside calibration.json. Migrating it to this same per-robot folder is a
follow-up; see docs/pd-controller-design.md.
"""

import json
import os
import time
from pathlib import Path

_LOG_ROOT = Path(os.path.normpath(os.path.join(
    os.path.dirname(__file__), os.pardir, os.pardir, os.pardir, os.pardir,
    "calibration_logs")))


def _team_key(team) -> str:
    return "yellow" if str(team).lower().startswith("y") else "blue"


def _split_stages(tried):
    """Split a flat tried-list back into coarse/fine stages.

    auto_tune_turn/auto_tune_linear always run a coarse grid followed by a
    same-size fine grid, so the list splits cleanly in half.
    """
    half = len(tried) // 2
    stages = []
    for name, chunk in (("coarse", tried[:half]), ("fine", tried[half:])):
        if not chunk:
            continue
        candidates = [
            {**gains, "score": result.score, "passed": result.passed}
            for gains, result in chunk
        ]
        winner_gains, winner_result = min(chunk, key=lambda t: t[1].score)
        stages.append({
            "name": name,
            "candidates": candidates,
            "winner": {**winner_gains, "score": winner_result.score},
        })
    return stages


def write_autotune_log(team, letter, shell_id, kind, result) -> Path:
    """Write one timestamped JSON file for an auto-tune sweep.

    Args:
        team: "yellow"/"blue" (or a truthy is_yellow-ish string).
        letter: robot letter, e.g. "A".
        shell_id: integer shell/robot id.
        kind: "turn" or "linear".
        result: an AutoTuneResult (pd_calibration.py) — uses .tried, .gains,
            .best_result, .log.

    Returns:
        Path to the written JSON file.
    """
    team_key = _team_key(team)
    robot_dir = _LOG_ROOT / team_key / str(letter).upper()
    robot_dir.mkdir(parents=True, exist_ok=True)

    timestamp = time.strftime("%Y-%m-%d_%H%M%S")
    file_path = robot_dir / f"{timestamp}_{kind}_autotune.json"

    payload = {
        "robot": {
            "team": team_key,
            "letter": str(letter).upper(),
            "shell_id": int(shell_id),
        },
        "kind": kind,
        "logged_at": timestamp,
        "stages": _split_stages(result.tried),
        "final_gains": result.gains,
        "final_score": result.best_result.score,
        "log": result.log,
    }

    with open(file_path, "w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")

    return file_path
