"""Unit tests for the noise-robust speedup scorer (optima/eval/scoring.py).

The whole point of this module is to make a sub-10% real win resolvable on a box
whose clocks can't be locked, and to refuse to crown on measurement noise. These
tests pin both halves: a genuine win passes, and noise alone never does.
"""

from optima.eval.scoring import relative_spread, score_speedup


def test_relative_spread_two_reads_is_range_over_mean():
    # The default bookend has exactly 2 baseline reads; spread is the honest gap.
    assert relative_spread([100.0, 110.0]) == (10.0 / 105.0)


def test_relative_spread_unmeasurable_below_two():
    assert relative_spread([100.0]) == float("inf")
    assert relative_spread([]) == float("inf")


def test_genuine_win_on_stable_box_passes():
    # Baselines agree (1% spread), candidate is a clean 12% faster -> real win.
    v = score_speedup([100.0, 101.0], 113.0, min_margin=0.02, k=2.0, max_noise=0.10)
    assert v.confident
    assert v.passed_speedup
    assert v.speedup > 1.11


def test_noise_alone_does_not_crown():
    # No real improvement (candidate ~= baseline mean) but candidate happens to read
    # a hair high; the noise-derived bar must reject it.
    v = score_speedup([100.0, 108.0], 106.0, min_margin=0.02, k=2.0, max_noise=0.10)
    # baseline spread = 8/104 ~= 7.7% -> required ~= 1 + 2*0.077 = 1.154; speedup ~1.019.
    assert not v.passed_speedup
    assert v.required > 1.15


def test_too_noisy_is_no_decision_not_a_pass():
    # Bracketing baselines disagree by >max_noise: untrustworthy round, never crown,
    # even if the raw ratio looks huge.
    v = score_speedup([100.0, 140.0], 150.0, max_noise=0.10)
    assert not v.confident
    assert not v.passed_speedup
    assert "NO-DECISION" in v.detail


def test_single_baseline_cannot_be_confident():
    # The legacy 2-launch shape (one baseline) can't measure noise -> not crownable.
    v = score_speedup([100.0], 130.0)
    assert not v.confident
    assert not v.passed_speedup
    assert "single baseline" in v.detail


def test_min_margin_floor_applies_on_a_perfectly_stable_box():
    # Zero measured noise still requires clearing the floor margin.
    v = score_speedup([100.0, 100.0], 101.0, min_margin=0.02, k=2.0)
    assert v.noise == 0.0
    assert v.required == 1.02
    assert not v.passed_speedup  # 1.01 < 1.02
    v2 = score_speedup([100.0, 100.0], 103.0, min_margin=0.02, k=2.0)
    assert v2.passed_speedup  # 1.03 >= 1.02 on a stable box


def test_a_real_loss_is_a_loss_not_no_decision():
    v = score_speedup([100.0, 101.0], 90.0, max_noise=0.10)
    assert v.confident  # the box was stable; we trust the verdict
    assert not v.passed_speedup
    assert v.speedup < 1.0
