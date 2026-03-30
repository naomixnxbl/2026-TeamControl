import math
import sys

from TeamControl.robot.Movement import calculateBallVelocity

def run_test(name, fn):
    try:
        fn()
        print(f"[PASS] {name}")
    except AssertionError as e:
        print(f"[FAIL] {name}: {e}")

def test_unreachable():
    beh = calculateBallVelocity(time_threshold=0.05)
    pose = (0.0, 0.0, 0.0)
    ball = (1.0, 0.0)
    dist, speed = beh.step(pose, ball)
    # distance must be ~1.0
    assert math.isclose(dist, 1.0, rel_tol=1e-6), f"dist={dist}, expected ≈1.0"
    # speed unreachable
    assert speed is None, f"speed={speed}, expected None"

def test_reachable_min_speed():
    beh = calculateBallVelocity(time_threshold=10.0)
    pose = (0.0, 0.0, 0.0)
    ball = (1.0, 0.0)
    dist, speed = beh.step(pose, ball)
    assert math.isclose(dist, 1.0, rel_tol=1e-6), f"dist={dist}, expected ≈1.0"
    # only fastest speed 0.10 finishes in <10s
    assert speed == 0.10, f"speed={speed}, expected 0.10"

def test_zero_distance():
    beh = calculateBallVelocity(time_threshold=1.0)
    pose = (2.0, 3.0, 0.0)
    ball = (2.0, 3.0)
    dist, speed = beh.step(pose, ball)
    assert math.isclose(dist, 0.0, abs_tol=1e-9), f"dist={dist}, expected 0.0"
    # zero distance means time=0 for all speeds, so picks slowest
    assert speed == 0.02, f"speed={speed}, expected 0.02"

def test_exact_threshold():
    beh = calculateBallVelocity(time_threshold=1.0)
    v = beh.speed_levels[2]    # e.g. 0.06
    d = v * beh.time_threshold
    pose = (0.0, 0.0, 0.0)
    ball = (d, 0.0)
    dist, speed = beh.step(pose, ball)
    # exactly at threshold, speed should match v
    assert math.isclose(speed, v, rel_tol=1e-6), f"speed={speed}, expected {v}"

def test_multiple_valid_choose_min():
    beh = calculateBallVelocity(time_threshold=1.0)
    pose = (0.0, 0.0, 0.0)
    ball = (0.08, 0.0)
    dist, speed = beh.step(pose, ball)
    # valid speeds [0.08,0.10] -> picks 0.08
    assert math.isclose(speed, 0.08, rel_tol=1e-6), f"speed={speed}, expected 0.08"

if __name__ == "__main__":
    tests = [
        test_unreachable,
        test_reachable_min_speed,
        test_zero_distance,
        test_exact_threshold,
        test_multiple_valid_choose_min,
    ]
    for t in tests:
        run_test(t.__name__, t)
    sys.exit(0)