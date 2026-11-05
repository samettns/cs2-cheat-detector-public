"""Demo düzeyi dağılım uyarısı (src/distribution_check.py) testleri.

- dondurulmuş referans: 160 CS2CD temiz eğitim maçı, ön kayıttaki yüzdelikler (iç %2.5–%97.5, dış %0.5–%99.5)
- seviye kuralı (ön kayıt): 2 = bir metrik dış aralık dışı VEYA iki metrik iç aralık dışı; 1 = tek metrik iç dışı
- M2: yalnız otomatik silah, < 20 dolu öldürme → hesaplanamadı
- uyarı skorları DEĞİŞTİRMEZ: referans varken/yokken Faz 2 skor tabloları birebir aynı
- ham CS2CD gerekmez: yalnız qc.json + features.parquet + dondurulmuş JSON
"""
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl
from polars.testing import assert_frame_equal

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))
import distribution_check as DC  # noqa: E402
import score_demo as S  # noqa: E402
from test_score_demo import make_work  # noqa: E402

REF = DC.load_reference()


def test_reference_frozen_from_clean_training_matches():
    pairs = pl.concat([pl.read_csv(f) for f in DC.PAIRS])
    per = pl.read_parquet(DC.REF_MATCHES)
    assert sorted(per["match"].to_list()) == sorted(pairs["clean"].to_list())       # yalnız temiz eğitim maçları
    assert all(m.startswith("no_cheater_present/") for m in per["match"].to_list())
    assert REF["inner_pct"] == [2.5, 97.5] and REF["outer_pct"] == [0.5, 99.5]
    for k in DC.METRICS:
        v = per[k].drop_nulls().to_numpy()
        r = REF["metrics"][k]
        assert r["n"] == len(v)
        assert r["inner"] == [round(float(np.percentile(v, p)), 3) for p in (2.5, 97.5)]
        assert r["outer"] == [round(float(np.percentile(v, p)), 3) for p in (0.5, 99.5)]
    assert REF["metrics"]["aim_p99"]["n"] == 160


def _vals(aim="mid", rea="mid"):
    def pick(k, where):
        r = REF["metrics"][k]
        return dict(mid=r["median"], soft_hi=(r["inner"][1] + r["outer"][1]) / 2, soft_lo=(r["inner"][0] + r["outer"][0]) / 2,
                    out_hi=r["outer"][1] + 1, out_lo=r["outer"][0] - 1, none=None)[where]
    return {"aim_p99": pick("aim_p99", aim), "reaction_med": pick("reaction_med", rea)}


def test_levels_follow_preregistered_rule():
    assert DC.classify(_vals(), REF)["level"] == 0
    assert DC.classify(_vals(aim="soft_hi"), REF)["level"] == 1
    assert DC.classify(_vals(rea="soft_lo"), REF)["level"] == 1
    assert DC.classify(_vals(aim="soft_hi", rea="soft_lo"), REF)["level"] == 2          # ikisi birlikte
    assert DC.classify(_vals(aim="out_hi"), REF)["level"] == 2
    assert DC.classify(_vals(rea="out_lo"), REF)["level"] == 2
    assert DC.classify(_vals(aim="soft_hi", rea="none"), REF)["level"] == 1              # M2 yoksa yalnız M1
    assert DC.classify(_vals(aim="none", rea="none"), REF)["level"] == 0
    r = DC.classify(_vals(rea="soft_lo"), REF)
    m = {d["metric"]: d for d in r["metrics"]}
    assert m["reaction_med"]["direction"] == "low" and m["reaction_med"]["explain"] == "daha kısa tepki süresi"
    assert m["aim_p99"]["direction"] is None


def test_texts_are_demo_level_and_not_alarm_language():
    for lv in (0, 1, 2):
        r = DC.classify(_vals(), REF) | dict(level=lv, title=DC.TITLE[lv], text=DC.TEXT[lv])
        txt = (r["title"] + r["text"] + r["note"] + DC.markdown(r)).lower()
        assert not any(w in txt for w in ("hileci", "şüphe", "alarm!", "tehlike"))
        assert "demo düzeyindedir" in txt and "professional/high-skill false-positive risk" in txt
        assert r["scope"] == "demo" and r["affects_scores"] is False
    assert DC.TEXT[2].startswith("Bu demonun davranış dağılımı Model v1'in temiz eğitim referansından farklı")
    assert DC.TEXT[1].startswith("Dağılım farkı gözlendi")


def test_reaction_median_only_automatic_and_min_kills():
    k = pl.DataFrame({"weapon_class": ["otomatik"] * 19 + ["sniper"] * 50,
                      "reaction_ms": [400.0] * 19 + [100.0] * 50})
    assert DC.reaction_median(k) == (None, 19)
    k2 = pl.concat([k, pl.DataFrame({"weapon_class": ["otomatik", "otomatik"], "reaction_ms": [600.0, None]})])
    v, n = DC.reaction_median(k2)
    assert n == 20 and v == 400.0                     # sniper öldürmeleri ve NaN dışarıda


def test_check_needs_only_analysis_folder(tmp_path):
    work = tmp_path / "a"
    work.mkdir()
    (work / "qc.json").write_text(json.dumps({"aim_step_p99": REF["metrics"]["aim_p99"]["outer"][1] + 2}), encoding="utf-8")
    pl.DataFrame({"weapon_class": ["otomatik"] * 30, "reaction_ms": [500.0] * 30}).write_parquet(work / "features.parquet")
    r = DC.check_analysis(work)
    assert r["level"] == 2 and r["reaction_n"] == 30


def test_scores_identical_with_and_without_reference(tmp_path, monkeypatch):
    a = make_work(tmp_path / "with", seed=3)
    b = make_work(tmp_path / "without", seed=3)
    S.score(a)
    monkeypatch.setattr(DC, "REF_FILE", tmp_path / "yok.json")
    S.score(b)
    assert (a / "distribution_check.json").exists() and not (b / "distribution_check.json").exists()
    for f in ["player_scores.parquet", "event_scores.parquet", "map_events.parquet"]:
        assert_frame_equal(pl.read_parquet(a / f), pl.read_parquet(b / f), check_exact=True)
    ra, rb = (json.loads((w / "model_result.json").read_text(encoding="utf-8")) for w in (a, b))
    ra.pop("seconds"), rb.pop("seconds")
    assert ra == rb
