from TeamControl.world.snapshot import (
    BallSnapshot,
    WorldSnapshot,
    empty_robot_team,
    snapshot_from_dict,
    snapshot_to_dict,
)


def make_snapshot():
    return WorldSnapshot(
        version=1,
        timestamp=10.0,
        frame_number=7,
        ball=BallSnapshot(1.0, 2.0),
        yellow=empty_robot_team(),
        blue=empty_robot_team(),
        us_yellow=True,
        us_positive=True,
        ball_candidates=(
            BallSnapshot(1.0, 2.0, confidence=0.8),
            BallSnapshot(3.0, 4.0, confidence=0.6),
        ),
    )


def test_snapshot_round_trip_preserves_ball_candidates():
    snapshot = make_snapshot()

    restored = snapshot_from_dict(snapshot_to_dict(snapshot))

    assert restored == snapshot


def test_snapshot_from_dict_supports_recordings_without_ball_candidates():
    payload = snapshot_to_dict(make_snapshot())
    payload.pop("ball_candidates")

    restored = snapshot_from_dict(payload)

    assert restored.ball_candidates == ()
