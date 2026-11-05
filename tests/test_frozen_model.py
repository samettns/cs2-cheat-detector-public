"""Dondurulmuş model v1 bütünlük testleri.  python -m pytest tests"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
V1 = Path("models/v1")
pytestmark = pytest.mark.skipif(not (V1 / "model_B.joblib").exists(), reason="models/v1 yok")


def test_feature_list_matches_model_b():
    import wallhack_eval as W
    meta = json.load(open(V1 / "features.json", encoding="utf-8"))
    assert meta["features"] == W.MODELS["B"][0]


def test_model_loads_and_has_15_inputs():
    import joblib
    m = joblib.load(V1 / "model_B.joblib")
    assert m.n_features_in_ == 15


def test_thresholds_present():
    t = json.load(open(V1 / "thresholds.json", encoding="utf-8"))
    assert 0 < t["G0"] < 1 and set(t["G1"]) == {"sniper ağırlıklı", "otomatik ağırlıklı", "diğer"}
