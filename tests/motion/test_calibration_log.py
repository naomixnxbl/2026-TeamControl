"""Unit tests for calibration_log.py — per-robot auto-tune history files."""

import json
from types import SimpleNamespace

from TeamControl.robot.motion import calibration_log


def _stub_result():
    """A fake AutoTuneResult: 3 distinct candidates repeated to fill two
    9-entry stages (matching the real 3x3 coarse + 3x3 fine grids)."""
    candidates = [
        ({"turn_kp": 0.5, "turn_kd": 0.0}, SimpleNamespace(score=10.0, passed=False)),
        ({"turn_kp": 1.0, "turn_kd": 0.1}, SimpleNamespace(score=2.0, passed=True)),
        ({"turn_kp": 1.5, "turn_kd": 0.2}, SimpleNamespace(score=8.0, passed=False)),
    ] * 3
    tried = candidates + candidates
    return SimpleNamespace(
        gains={"turn_kp": 1.0, "turn_kd": 0.1},
        best_result=SimpleNamespace(score=2.0),
        tried=tried,
        log=["line 1", "line 2"],
    )


def test_writes_json_under_team_letter_folder(tmp_path, monkeypatch):
    monkeypatch.setattr(calibration_log, "_LOG_ROOT", tmp_path / "calibration_logs")

    path = calibration_log.write_autotune_log(
        team="yellow", letter="A", shell_id=0, kind="turn", result=_stub_result(),
    )

    assert path.exists()
    assert path.parent == tmp_path / "calibration_logs" / "yellow" / "A"
    assert path.name.endswith("_turn_autotune.json")


def test_payload_shape(tmp_path, monkeypatch):
    monkeypatch.setattr(calibration_log, "_LOG_ROOT", tmp_path / "calibration_logs")

    path = calibration_log.write_autotune_log(
        team="blue", letter="b", shell_id=3, kind="linear", result=_stub_result(),
    )

    payload = json.loads(path.read_text())
    assert payload["robot"] == {"team": "blue", "letter": "B", "shell_id": 3}
    assert payload["kind"] == "linear"
    assert payload["final_gains"] == {"turn_kp": 1.0, "turn_kd": 0.1}
    assert payload["final_score"] == 2.0
    assert [s["name"] for s in payload["stages"]] == ["coarse", "fine"]
    assert len(payload["stages"][0]["candidates"]) == 9
    assert payload["stages"][0]["winner"]["score"] == 2.0
