"""Görünürlük yardımcıları testleri.  Çalıştır: python -m pytest tests"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from visibility import EYE_CROUCH, EYE_STAND, eye_height, is_masked  # noqa: E402


@pytest.mark.parametrize("a, b", [("Bridge", "Canal"), ("Canal", "Bridge")])
def test_anubis_bridge_canal_masked_both_directions(a, b):
    assert is_masked("de_anubis", a, b)


@pytest.mark.parametrize("map_name, a, b", [
    ("de_anubis", "Bridge", "Bridge"),   # aynı bölge maskeli değil
    ("de_anubis", "Canal", "Middle"),
    ("de_anubis", None, "Canal"),        # bölge bilinmiyorsa maskeleme yok
    ("de_mirage", "Bridge", "Canal"),    # kural sadece anubis için
])
def test_not_masked(map_name, a, b):
    assert not is_masked(map_name, a, b)


def test_eye_height_interpolates():
    assert eye_height(0) == EYE_STAND
    assert eye_height(1) == EYE_CROUCH
    assert eye_height(None) == EYE_STAND
    assert eye_height(0.5) == pytest.approx((EYE_STAND + EYE_CROUCH) / 2)
