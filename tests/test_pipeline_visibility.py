"""Görünürlük tabanlı özelliklerin features.py hattından modele ulaştığını doğrular.  python -m pytest tests"""
import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from baseline import AIM_FEATURES, VISIBILITY_FEATURES, WEAPON_FEATURES  # noqa: E402

KF = Path("data/processed/kill_features.parquet")
WF = Path("data/processed/wallhack_features.parquet")
needs_data = pytest.mark.skipif(not (KF.exists() and WF.exists() and Path("data/pairs.csv").exists()),
                                reason="işlenmiş veri yok")


def test_visibility_list_is_the_preregistered_one():
    assert VISIBILITY_FEATURES == ["inv_angle_med", "inv_frac_lt10", "prefov_angle", "wallbang", "reaction_ms"]


def test_model_b_uses_visibility_and_a_does_not():
    import wallhack_eval as W
    a, b = W.MODELS["A"][0], W.MODELS["B"][0]
    assert a == AIM_FEATURES + WEAPON_FEATURES
    assert b == AIM_FEATURES + WEAPON_FEATURES + VISIBILITY_FEATURES
    assert W.MODELS["B"][1] == []          # B: native NaN, doldurma yok


@needs_data
def test_kill_features_contains_visibility_columns():
    cols = pl.read_parquet_schema(KF)
    assert all(c in cols for c in VISIBILITY_FEATURES)


@needs_data
def test_visibility_values_reach_model_matrix():
    import wallhack_eval as W
    df = W.dataset(False)
    assert not [c for c in df.columns if c.endswith("_right")], "çift sütun: görünürlük iki kez eklenmiş"
    feats = W.MODELS["B"][0]
    X = df.select(feats).to_numpy().astype(float)
    for c in VISIBILITY_FEATURES:
        col = X[:, feats.index(c)]
        assert np.isfinite(col).sum() > 0.5 * len(col), f"{c} modele ulaşmıyor (çoğu NaN)"


@needs_data
def test_pipeline_values_equal_source_table():
    kf = pl.read_parquet(KF).select("match", "player", "tick", *VISIBILITY_FEATURES)
    wf = (pl.read_parquet(WF).select("match", "player", pl.col("kill_tick").alias("tick"), *VISIBILITY_FEATURES))
    j = wf.join(kf, on=["match", "player", "tick"], suffix="_kf")
    assert j.height >= wf.height
    for c in VISIBILITY_FEATURES:
        assert (j[c].is_null() == j[f"{c}_kf"].is_null()).all()
        assert (j[c].drop_nulls() == j[f"{c}_kf"].drop_nulls()).all()
