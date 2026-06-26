"""
Unit tests for PDController (robot/motion/pd.py).

Each test exercises one specific behaviour so that if something breaks,
the test name tells you exactly what is wrong.
"""
import math

import pytest

from TeamControl.robot.motion.pd import PDController


class TestScalarPD:
    """Tests where the error is a single number (e.g. angle in radians)."""

    def test_first_tick_is_p_only(self):
        # On the very first call there is no previous sample, so de/dt = 0.
        # Output must equal Kp * error with no D contribution.
        pd = PDController(kp=2.0, kd=5.0)
        out = pd.update(5.0, now=0.0)
        assert out == pytest.approx(10.0)  # 2.0 * 5.0

    def test_second_tick_includes_d_term(self):
        # error goes 10 → 8 over 1 second, so de/dt = -2.
        # P = 2*8 = 16, D = 1*(-2) = -2, total = 14.
        pd = PDController(kp=2.0, kd=1.0)
        pd.update(10.0, now=0.0)
        out = pd.update(8.0, now=1.0)
        assert out == pytest.approx(14.0)

    def test_d_term_damps_when_error_shrinks(self):
        # D term should oppose P when the error is shrinking.
        pd = PDController(kp=1.0, kd=1.0)
        pd.update(10.0, now=0.0)
        out_with_d = pd.update(5.0, now=1.0)  # de/dt = -5

        pd2 = PDController(kp=1.0, kd=0.0)
        pd2.update(10.0, now=0.0)
        out_p_only = pd2.update(5.0, now=1.0)

        assert out_with_d < out_p_only  # D brakes, so output is smaller

    def test_output_saturates_positive(self):
        pd = PDController(kp=2.0, kd=0.0, out_limit=5.0)
        out = pd.update(100.0, now=0.0)
        assert out == pytest.approx(5.0)

    def test_output_saturates_negative(self):
        pd = PDController(kp=2.0, kd=0.0, out_limit=5.0)
        out = pd.update(-100.0, now=0.0)
        assert out == pytest.approx(-5.0)

    def test_no_limit_does_not_clip(self):
        pd = PDController(kp=2.0, kd=0.0, out_limit=None)
        out = pd.update(100.0, now=0.0)
        assert out == pytest.approx(200.0)

    def test_reset_makes_next_tick_p_only(self):
        # After reset the controller forgets history, so the next call has D=0.
        pd = PDController(kp=2.0, kd=5.0)
        pd.update(10.0, now=0.0)
        pd.reset()
        out = pd.update(5.0, now=1.0)
        assert out == pytest.approx(10.0)  # 2.0 * 5.0

    def test_zero_dt_guard_does_not_raise(self):
        # Two calls with the same timestamp should not cause a ZeroDivisionError.
        pd = PDController(kp=1.0, kd=1.0)
        pd.update(5.0, now=1.0)
        out = pd.update(3.0, now=1.0)  # dt clamped to 1e-6
        assert math.isfinite(out)

    def test_negative_error_produces_negative_output(self):
        pd = PDController(kp=2.0, kd=0.0)
        out = pd.update(-4.0, now=0.0)
        assert out == pytest.approx(-8.0)


class TestVectorPD:
    """Tests where the error is a 2-element tuple (e.g. (ex, ey) in mm)."""

    def test_first_tick_is_p_only(self):
        pd = PDController(kp=1.0, kd=0.0)
        out = pd.update((3.0, 4.0), now=0.0)
        assert out == pytest.approx((3.0, 4.0))

    def test_magnitude_clamped_when_over_limit(self):
        pd = PDController(kp=1.0, kd=0.0, out_limit=5.0)
        out = pd.update((6.0, 8.0), now=0.0)  # raw magnitude = 10
        mag = math.sqrt(out[0] ** 2 + out[1] ** 2)
        assert mag == pytest.approx(5.0)

    def test_direction_preserved_on_clamp(self):
        # Scaling down should not change the direction (vy/vx ratio).
        pd = PDController(kp=1.0, kd=0.0, out_limit=5.0)
        out = pd.update((6.0, 8.0), now=0.0)
        assert out[1] / out[0] == pytest.approx(8.0 / 6.0)

    def test_no_clamp_when_under_limit(self):
        pd = PDController(kp=1.0, kd=0.0, out_limit=10.0)
        out = pd.update((3.0, 4.0), now=0.0)  # magnitude = 5, under limit
        assert out == pytest.approx((3.0, 4.0))

    def test_reset_restores_p_only_on_next_tick(self):
        pd = PDController(kp=1.0, kd=1.0)
        pd.update((10.0, 0.0), now=0.0)
        pd.reset()
        out = pd.update((5.0, 0.0), now=1.0)
        assert out == pytest.approx((5.0, 0.0))

    def test_d_term_opposes_shrinking_error(self):
        # Error shrinks: de/dt is negative, D term reduces output.
        pd = PDController(kp=1.0, kd=1.0)
        pd.update((10.0, 0.0), now=0.0)
        out_x, _ = pd.update((6.0, 0.0), now=1.0)  # de/dt = -4 on x

        pd2 = PDController(kp=1.0, kd=0.0)
        pd2.update((10.0, 0.0), now=0.0)
        out_p_only_x, _ = pd2.update((6.0, 0.0), now=1.0)

        assert out_x < out_p_only_x
