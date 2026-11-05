"""Veri sızıntısı (data leakage) kontrolleri.  Çalıştır: python -m pytest tests"""
import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
import baseline  # noqa: E402

# Modele asla girmemesi gereken sütunlar: etiket, kimlik, dosya/klasör kaynaklı bilgiler
FORBIDDEN = {"y", "group", "cheater", "cheaters", "player", "match", "steamid",
             "match_player", "tick", "weapon"}  # weapon yerine sadece silah sınıfı kullanılıyor

ALL_SETUPS = baseline.FEATURE_SETS


@pytest.mark.parametrize("name", list(ALL_SETUPS))
def test_no_forbidden_columns(name):
    assert not set(ALL_SETUPS[name]) & FORBIDDEN, f"{name}: yasak sütun kullanılıyor"


DATA = Path("data/processed/kill_features.parquet")


@pytest.mark.skipif(not DATA.exists(), reason="özellik tablosu yok")
def test_folds_do_not_share_matches():
    df = baseline.load()
    groups = df["match"].to_numpy()
    for train, test in GroupKFold(n_splits=5).split(np.zeros(len(groups)), groups=groups):
        assert not set(groups[train]) & set(groups[test]), "aynı maç hem eğitimde hem testte"


@pytest.mark.skipif(not DATA.exists(), reason="özellik tablosu yok")
def test_player_ids_unique_per_match():
    # Player_N isimleri maçlar arasında tekrarlanıyor: her player kimliği tek bir maça ait olmalı
    df = baseline.load()
    per_player = df.group_by("player").agg(pl.col("match").n_unique().alias("n"))
    assert per_player["n"].max() == 1
