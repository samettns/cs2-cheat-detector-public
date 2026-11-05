"""Faz 2 (src/score_demo.py) testleri.

Hızlı: sentetik Faz 1 klasörü (değerler eğitim referansından örneklenir) — özellik sırası, UYGUN DEĞİL → skor yok,
determinizm, oyuncu birleştirme (ortalama, >= 3 öldürme), G0/G1 eşikleri (skor > eşik), inceleme önceliği,
bıçak açıklaması, map_events alanları, Faz 1 dosyalarının değişmemesi, Türkçe yüzdelik eki.
Yavaş (RUN_SLOW=1): CS2CD gerileme (3 maç; Faz 1 + Faz 2 skorları = v1 değerlendirme kodu yolu, birebir) ve
gerçek demo çıktısında map_events eksiksizliği.
"""
import json
import os
import shutil
import sys
from pathlib import Path

import joblib
import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
import score_demo as S  # noqa: E402

SLOW = pytest.mark.skipif(not os.environ.get("RUN_SLOW"), reason="yavaş test; RUN_SLOW=1 ile çalıştır")
FEATS = json.loads(Path("models/v1/features.json").read_text(encoding="utf-8"))["features"]
MAP_EVENT_REQUIRED = ["map", "round", "tick", "kill_id", "event_type", "reason", "attacker", "victim", "event_score",
                      "attacker_X", "attacker_Y", "attacker_Z", "attacker_yaw", "attacker_pitch",
                      "victim_X", "victim_Y", "victim_Z", "wallbang", "thrusmoke", "headshot", "explanation"]
MAP_EVENT_NONNULL = ["map", "round", "tick", "kill_id", "event_type", "reason", "attacker", "victim", "event_score",
                     "attacker_X", "attacker_Y", "attacker_Z", "attacker_yaw", "attacker_pitch",
                     "victim_X", "victim_Y", "victim_Z", "wallbang", "thrusmoke", "headshot", "explanation"]


# ---------------------------------------------------------------- sentetik Faz 1 klasörü
def make_work(tmp: Path, applicable: bool = True, kills_per_player=(5, 4, 3, 2, 6), weapons=None, seed=0) -> Path:
    """Eğitim referansındaki temiz öldürmelerden örneklenmiş değerlerle Faz 1 çıktısı taklidi."""
    rng = np.random.default_rng(seed)
    ref = pl.read_parquet(S.REF / "clean_kill_features.parquet")
    rows, kills, kid = [], [], 0
    for p, n in enumerate(kills_per_player):
        for j in range(n):
            r = ref.row(int(rng.integers(ref.height)), named=True)
            wc = r["weapon_class"]
            weapon = (weapons or {}).get((p, j)) or {"otomatik": "ak47", "tabanca": "glock", "sniper": "awp"}[wc]
            tick = 1000 + 500 * kid
            f = {c: r[c] for c in S.load_frozen()["ref"]["explain_features"]}
            f["shots"] = int(f["shots"])
            rows.append(dict(kill_id=kid, match="x/x", player=f"Player_{p + 1}", tick=tick, weapon=weapon,
                             weapon_class=wc, **{c: f.get(c) for c in FEATS if c in f},
                             is_sniper=int(wc == "sniper"), is_pistol=int(wc == "tabanca"), wallbang=float(r["wallbang"])))
            kills.append(dict(kill_id=kid, round=1 + kid // 3, victim=f"Player_{6 + p % 5}", headshot=bool(kid % 2),
                              penetrated=int(r["wallbang"]), thrusmoke=False, pre_match=False, in_model=True,
                              **{f"{role}_{a}": float(rng.normal(0, 500)) if a in "XYZ" else float(rng.uniform(-89, 89))
                                 for role in ("attacker", "victim") for a in ("X", "Y", "Z", "yaw", "pitch")}))
            kid += 1
    feats = pl.DataFrame(rows).select(S.ID_COLS + FEATS).with_columns(
        pl.col("shots").cast(pl.Int64), pl.col("is_sniper").cast(pl.Int8), pl.col("is_pistol").cast(pl.Int8),
        pl.col("kill_id").cast(pl.UInt32))
    work = tmp / "demo"
    work.mkdir(parents=True, exist_ok=True)
    feats.write_parquet(work / "features.parquet")
    pl.DataFrame(kills).with_columns(pl.col("kill_id").cast(pl.UInt32)).write_parquet(work / "kills.parquet")
    qc = dict(demo_id="sentetik", model_v1_applicable=applicable, feature_order=FEATS,
              model_v1_reasons=[] if applicable else ["harita de_x desteklenmiyor (.tri yok)"])
    (work / "qc.json").write_text(json.dumps(qc), encoding="utf-8")
    (work / "metadata.json").write_text(json.dumps(dict(demo_id="sentetik", map="de_dust2", rounds=10, source="other")),
                                        encoding="utf-8")
    return work


def outputs(work: Path) -> dict:
    out = {f: pl.read_parquet(work / f) for f in ["player_scores.parquet", "event_scores.parquet", "map_events.parquet"]}
    res = json.loads((work / "model_result.json").read_text(encoding="utf-8"))
    res.pop("seconds")
    out["model_result.json"] = res
    out["model_report.md"] = (work / "model_report.md").read_text(encoding="utf-8")
    return out


# ---------------------------------------------------------------- hızlı testler
def test_model_expects_frozen_feature_order():
    fr = S.load_frozen()
    import baseline as B
    assert fr["features"] == FEATS == B.AIM_FEATURES + B.WEAPON_FEATURES + B.VISIBILITY_FEATURES
    assert fr["model"].n_features_in_ == 15


def test_wrong_feature_order_or_missing_feature_stops(tmp_path):
    work = make_work(tmp_path)
    f = pl.read_parquet(work / "features.parquet")
    f.select(S.ID_COLS + FEATS[::-1]).write_parquet(work / "features.parquet")
    with pytest.raises(S.ScoringError, match="sütun sırası"):
        S.score(work)
    f.drop("reaction_ms").write_parquet(work / "features.parquet")
    with pytest.raises(S.ScoringError, match="eksik özellik"):
        S.score(work)


def test_not_applicable_produces_no_scores(tmp_path):
    work = make_work(tmp_path, applicable=False)
    res = S.score(work)
    assert res["status"] == "skor üretilmedi"
    for f in ["player_scores.parquet", "event_scores.parquet", "map_events.parquet"]:
        assert not (work / f).exists()
    r = json.loads((work / "model_result.json").read_text(encoding="utf-8"))
    assert r["status"] == "skor üretilmedi" and "desteklenmiyor" in r["model_v1_reasons"][0]
    assert "Skor üretilmedi" in (work / "model_report.md").read_text(encoding="utf-8")


def test_determinism_same_phase1_same_scores(tmp_path):
    a = outputs((S.score(make_work(tmp_path / "a")), tmp_path / "a" / "demo")[1])
    b = outputs((S.score(make_work(tmp_path / "b")), tmp_path / "b" / "demo")[1])
    for k in a:
        if isinstance(a[k], pl.DataFrame):
            assert_frame_equal(a[k], b[k], check_exact=True)
        else:
            assert a[k] == b[k], k


def test_player_aggregation_mean_and_min_kills(tmp_path):
    work = make_work(tmp_path, kills_per_player=(5, 4, 3, 2, 1))
    S.score(work)
    p = pl.read_parquet(work / "player_scores.parquet")
    e = pl.read_parquet(work / "event_scores.parquet")
    means = e.group_by("attacker").agg(m=pl.col("event_score").mean(), n=pl.len())
    j = p.join(means, left_on="player", right_on="attacker")
    ok = j.filter(pl.col("n") >= 3)
    assert ok.height == 3 and np.allclose(ok["player_raw_score"].to_numpy(), ok["m"].to_numpy(), rtol=0, atol=0)
    few = j.filter(pl.col("n") < 3)
    assert few.height == 2 and few["player_raw_score"].is_null().all()
    assert (few["review_priority"] == "yetersiz veri").all() and few["style_percentile"].is_null().all()


def test_raw_scores_are_model_predictions(tmp_path):
    work = make_work(tmp_path)
    S.score(work)
    e = pl.read_parquet(work / "event_scores.parquet").sort("kill_id")
    f = pl.read_parquet(work / "features.parquet").sort("kill_id")
    m = joblib.load("models/v1/model_B.joblib")
    assert np.array_equal(e["event_score"].to_numpy(), m.predict_proba(f.select(FEATS).to_numpy().astype(float))[:, 1])


def test_g1_thresholds_match_clean_reference():
    fr = S.load_frozen()
    ref = fr["clean_ref"]
    for st, thr in fr["thresholds"]["G1"].items():
        assert thr == float(np.quantile(ref.filter(pl.col("stil") == st)["score"].to_numpy(), 0.95))
    assert fr["thresholds"]["G0"] == float(np.quantile(ref["score"].to_numpy(), 0.95))


def _three_kill_players(styles_scores):
    """Her oyuncu 3 öldürme; stil silah sınıfıyla belirlenir; öldürme skorları = oyuncu skoru (ortalama aynı)."""
    rows, scores = [], []
    wc_of = {"sniper ağırlıklı": "sniper", "otomatik ağırlıklı": "otomatik", "diğer": "tabanca"}
    for i, (st, s) in enumerate(styles_scores):
        for j in range(3):
            wc = wc_of[st]
            rows.append(dict(kill_id=3 * i + j, match="x/x", player=f"Player_{i + 1}", tick=j, weapon="w",
                             weapon_class=wc, is_sniper=int(wc == "sniper"), is_pistol=int(wc == "tabanca")))
            scores.append(s)
    return pl.DataFrame(rows), np.array(scores)


def test_g0_g1_and_priority_boundaries():
    fr = S.load_frozen()
    th = fr["thresholds"]
    st = "otomatik ağırlıklı"
    g1 = th["G1"][st]
    p90 = float(np.quantile(fr["clean_ref"].filter(pl.col("stil") == st)["score"].to_numpy(), 0.90))
    cases = [(st, g1 + 1e-9), (st, g1), ((st), (p90 + g1) / 2), (st, p90), (st, p90 - 1e-6)]
    df, sc = _three_kill_players(cases)
    t = S.player_table(df, sc, fr).sort("player")
    pr = t["review_priority"].to_list()
    assert pr == ["yüksek inceleme önceliği", "orta inceleme önceliği", "orta inceleme önceliği",
                  "düşük inceleme önceliği", "düşük inceleme önceliği"]
    assert t["g1_above_style_threshold"].to_list() == [True, False, False, False, False]   # skor > eşik (kesin)
    assert (t["style"] == st).all()
    # G0 yalnız karşılaştırma: aynı skorlarda G0 kararı ayrı sütunda
    df, sc = _three_kill_players([("sniper ağırlıklı", th["G0"] + 1e-9), ("sniper ağırlıklı", th["G0"])])
    t = S.player_table(df, sc, fr).sort("player")
    assert t["g0_above_threshold"].to_list() == [True, False]
    # sniper: G0 üstü ama G1 (0.863) altı → yüksek değil
    assert t["review_priority"][0] != "yüksek inceleme önceliği" and t["low_confidence_style"].all()


def test_style_percentile_definition():
    fr = S.load_frozen()
    st = "diğer"
    ref = fr["clean_ref"].filter(pl.col("stil") == st)["score"].to_numpy()
    s = float(np.median(ref))
    df, sc = _three_kill_players([(st, s)])
    t = S.player_table(df, sc, fr)
    assert t["style_percentile"][0] == round(float((ref < s).mean()) * 100, 1)


def test_knife_kill_gets_no_automatic_weapon_explanation(tmp_path):
    work = make_work(tmp_path, kills_per_player=(4,), weapons={(0, 0): "knife_gut"})
    f = pl.read_parquet(work / "features.parquet").with_columns(
        weapon_class=pl.when(pl.col("weapon") == "knife_gut").then(pl.lit("otomatik")).otherwise(pl.col("weapon_class")))
    f.write_parquet(work / "features.parquet")
    S.score(work)
    e = pl.read_parquet(work / "event_scores.parquet")
    k = e.filter(pl.col("weapon") == "knife_gut")
    assert k["is_knife"][0] and "bıçak" in k["explanation"][0] and "otomatik silah" not in k["explanation"][0]
    assert k["top_features"][0] == "[]"


def test_map_events_fields_complete(tmp_path):
    work = make_work(tmp_path)
    S.score(work)
    m = pl.read_parquet(work / "map_events.parquet")
    assert all(c in m.columns for c in MAP_EVENT_REQUIRED)
    assert m.height > 0 and all(m[c].null_count() == 0 for c in MAP_EVENT_NONNULL)
    e = pl.read_parquet(work / "event_scores.parquet")
    assert set(e.filter(pl.col("review_point"))["kill_id"].to_list()) <= set(m["kill_id"].to_list())
    assert m["attacker"].str.starts_with("Player_").all() and m["victim"].str.starts_with("Player_").all()


def test_phase1_files_not_modified_and_disclaimer(tmp_path):
    work = make_work(tmp_path)
    before = {f.name: f.read_bytes() for f in work.iterdir() if f.is_file()}
    S.score(work)
    for name, data in before.items():
        assert (work / name).read_bytes() == data, name
    rep = (work / "model_report.md").read_text(encoding="utf-8")
    res = json.loads((work / "model_result.json").read_text(encoding="utf-8"))
    assert S.DISCLAIMER in rep and res["disclaimer"] == S.DISCLAIMER and "kanıtı" in rep
    assert "hile ihtimali" not in rep and "olasılığı %" not in rep and "hileci" not in rep.replace("'hileci'", "")


def test_turkish_percent_suffix():
    assert [S.pct_from(p) for p in (98, 90, 62, 100, 5, 6, 3, 40, 0, 73, 99)] == [
        "%98'inden", "%90'ından", "%62'sinden", "%100'ünden", "%5'inden", "%6'sından", "%3'ünden", "%40'ından",
        "%0'ından", "%73'ünden", "%99'undan"]


# ---------------------------------------------------------------- yavaş: CS2CD gerileme + gerçek demo
REGRESSION = ["no_cheater_present/108", "no_cheater_present/67", "no_cheater_present/88"]


@SLOW
@pytest.mark.parametrize("match", REGRESSION)
def test_cs2cd_phase1_plus_phase2_equals_v1_evaluation_path(tmp_path, match):
    """Faz 1 + Faz 2 öldürme ve oyuncu skorları = v1 değerlendirme yolu (W.dataset → model_B → W.player_scores).
    Not: bu maçlar eğitimdeydi; skorlar in-sample (yalnız hat doğruluğu sınanır, performans değil)."""
    import analyze_demo as AD
    import wallhack_eval as W
    from pooled import PAIRS_ALL
    work = AD.analyze(cs2cd=match, out_root=tmp_path)
    S.score(work)
    model = joblib.load("models/v1/model_B.joblib")
    df = W.dataset(pairs_files=PAIRS_ALL).filter(pl.col("match") == match)
    s = model.predict_proba(df.select(FEATS).to_numpy().astype(float))[:, 1]
    short = pl.lit("Player_") + pl.col("player").str.split("_").list.last()
    old_k = df.select(player=short, tick="tick").with_columns(score=pl.Series(s)).sort("player", "tick", "score")
    new_k = (pl.read_parquet(work / "event_scores.parquet")
               .select(player="attacker", tick="kill_tick", score="event_score").sort("player", "tick", "score"))
    assert_frame_equal(old_k, new_k, check_exact=True)
    old_p = W.player_scores(df, s).select(player=short, score="score").sort("player")
    new_p = (pl.read_parquet(work / "player_scores.parquet").drop_nulls("player_raw_score")
               .select("player", score="player_raw_score").sort("player"))
    assert_frame_equal(old_p, new_p, check_exact=True)


REAL = Path("reports/analysis")


@SLOW
@pytest.mark.skipif(not any(REAL.glob("*/map_events.parquet")), reason="Faz 2 gerçek demo çıktısı yok")
def test_real_demo_map_events_complete():
    for f in REAL.glob("*/map_events.parquet"):
        m = pl.read_parquet(f)
        assert all(c in m.columns for c in MAP_EVENT_REQUIRED), f
        assert all(m[c].null_count() == 0 for c in MAP_EVENT_NONNULL), f
        p = pl.read_parquet(f.parent / "player_scores.parquet")
        assert p["player"].str.starts_with("Player_").all()
