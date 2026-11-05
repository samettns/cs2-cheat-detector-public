"""Wallhack özellik tanımlarının sentetik veriyle testleri.  Çalıştır: python -m pytest tests"""
import sys
from pathlib import Path

import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from wallhack_features import invisible_tracking, onset_features  # noqa: E402

M, P, K = "no_cheater_present/1", "no_cheater_present/1_Player_3", 100   # maç, oyuncu, öldürme tick'i


def victim_window(visible_from=60, masked_ticks=()):
    """Kurban 0..58 arası görünmez, visible_from'dan itibaren görünür; açı = tick (kolay kontrol için)."""
    rows = []
    for t in range(0, K + 1, 2):
        vis = None if t in masked_ticks else (t >= visible_from)
        rows.append(dict(match=M, player=P, kill_tick=K, tick=t, enemy="Player_7", is_victim=True,
                         visible=vis, spotted=False, angle=float(t)))
    return pl.DataFrame(rows)


def test_onset_prefov_and_reaction():
    hurts = {(M, "Player_3", "Player_7"): [40, 76, 100]}   # 40 < t_vis olduğu için sayılmamalı
    out = onset_features(victim_window(), deaths={}, hurts=hurts).row(0, named=True)
    assert out["t_vis"] == 60
    assert out["prefov_angle"] == 48.0                      # t_vis − 12 anındaki açı
    assert out["reaction_ms"] == pytest.approx((76 - 60) / 64 * 1000)
    assert out["f2_nan"] is None


def test_onset_smoke_is_nan():
    deaths = {(M, K, "Player_3"): {"thrusmoke": True}}
    out = onset_features(victim_window(), deaths=deaths, hurts={}).row(0, named=True)
    assert out["prefov_angle"] is None and out["f2_nan"] == "duman"


def test_onset_mask_in_prefov_range_is_nan():
    out = onset_features(victim_window(masked_ticks=(54,)), deaths={}, hurts={}).row(0, named=True)
    assert out["prefov_angle"] is None and out["f2_nan"] == "maske"


def test_onset_no_transition():
    out = onset_features(victim_window(visible_from=0), deaths={}, hurts={}).row(0, named=True)
    assert out["f2_nan"] == "gecis_yok"


def test_invisible_tracking_uses_min_angle_over_invisible_enemies():
    rows = []
    for t in range(0, 40, 2):   # 20 an
        rows += [dict(match=M, player=P, kill_tick=K, tick=t, enemy="A", is_victim=False, visible=False,
                      spotted=False, angle=5.0),
                 dict(match=M, player=P, kill_tick=K, tick=t, enemy="B", is_victim=False, visible=False,
                      spotted=False, angle=50.0),
                 dict(match=M, player=P, kill_tick=K, tick=t, enemy="C", is_victim=False, visible=True,
                      spotted=False, angle=1.0)]   # görünür rakip sayılmamalı
    out = invisible_tracking(pl.DataFrame(rows)).row(0, named=True)
    assert out["inv_angle_med"] == 5.0 and out["inv_frac_lt10"] == 1.0 and out["f1_nan"] is None


def test_invisible_tracking_too_few_ticks_is_nan():
    rows = [dict(match=M, player=P, kill_tick=K, tick=t, enemy="A", is_victim=False, visible=False,
                 spotted=False, angle=5.0) for t in range(0, 10, 2)]   # 5 an < 10
    out = invisible_tracking(pl.DataFrame(rows)).row(0, named=True)
    assert out["inv_angle_med"] is None and out["f1_nan"] == "az_gorunmez_an"
