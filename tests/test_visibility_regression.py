"""Görünürlük hesabı gerileme testi: optimize edilmiş match_windows, kayıtlı (optimizasyon öncesi) sonuçlarla
birebir aynı mı? 3 eski maç: anubis (maske), dumanlı öldürmeler, round başında öldürme (pencere round öncesine taşar).

Yavaş (her maç ~1 dk ray-cast) → varsayılan olarak atlanır. Çalıştırmak için:
    RUN_SLOW=1 python -m pytest tests/test_visibility_regression.py
"""
import os
import sys
from pathlib import Path

import polars as pl
import pytest
from polars.testing import assert_frame_equal

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

SNAPSHOT = Path("data/processed/snapshot_before_expansion/visibility.parquet")
CASES = {
    "anubis, maske devrede": "no_cheater_present/108",
    "dumanlı öldürmeler": "no_cheater_present/67",
    "round başında öldürme": "no_cheater_present/88",
}
SORT = ["match", "player", "kill_tick", "tick", "enemy", "is_victim"]

pytestmark = [
    pytest.mark.skipif(not os.environ.get("RUN_SLOW"), reason="yavaş test; RUN_SLOW=1 ile çalıştır"),
    pytest.mark.skipif(not SNAPSHOT.exists(), reason="kayıtlı sonuç yok"),
]


@pytest.mark.parametrize("case, match", list(CASES.items()))
def test_match_windows_equals_saved(case, match):
    from visibility import match_windows

    new = match_windows(match).sort(SORT)
    old = pl.read_parquet(SNAPSHOT).filter(pl.col("match") == match).sort(SORT)
    assert new.height == old.height > 0, f"{case}: satır sayısı farklı ({new.height} vs {old.height})"
    # Birebir eşitlik (float toleransı yok): aynı girdi, aynı işlem sırası
    assert_frame_equal(new.select(old.columns), old, check_exact=True)
    # NaN/null konumları (maskeli görünürlük) aynı
    assert (new["visible"].is_null() == old["visible"].is_null()).all()
