"""Yaw sarma (wrap-around) düzeltmesi testleri.  Çalıştır: python -m pytest tests"""
import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from aim import TICKRATE, add_aim_speed, angle_diff  # noqa: E402
from features import player_arrays  # noqa: E402


@pytest.mark.parametrize("a, b, expected", [
    (-179, 179, 2),     # 179 -> -179: sağa 2 derece, 358 değil
    (179, -179, -2),    # ters yön
    (10, 350, 20),
    (0, 0, 0),
    (90, -90, -180),    # tam ters: -180 (veya +180) kabul
])
def test_angle_diff_wraps(a, b, expected):
    assert angle_diff(a, b) == pytest.approx(expected)


def test_angle_diff_always_in_range():
    rng = np.random.default_rng(0)
    a, b = rng.uniform(-180, 180, 10_000), rng.uniform(-180, 180, 10_000)
    d = angle_diff(a, b)
    assert (d >= -180).all() and (d < 180).all()


def _ticks(yaws):
    n = len(yaws)
    return pl.DataFrame({"tick": list(range(n)), "steamid": ["P"] * n,
                         "yaw": yaws, "pitch": [0.0] * n, "is_alive": [True] * n})


def test_add_aim_speed_across_wrap():
    # 179 -> -179 -> -177: her adım 2 derece = 128 °/s
    out = add_aim_speed(_ticks([179.0, -179.0, -177.0]))
    assert out["aim_speed"].to_list()[1:] == pytest.approx([2 * TICKRATE, 2 * TICKRATE])


def test_player_arrays_speed_across_wrap():
    # features.py'nin kullandığı yol da aynı düzeltmeyi uyguluyor mu?
    arr = player_arrays(_ticks([179.0, -179.0, -177.0]), t0=0, n=3)["P"]
    assert arr["speed"][1:] == pytest.approx([2 * TICKRATE, 2 * TICKRATE])
