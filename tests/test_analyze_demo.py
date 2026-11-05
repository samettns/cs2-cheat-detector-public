"""Faz 1 analiz hattı (src/analyze_demo.py) testleri.

Hızlı (her zaman): ön kontroller (tick hızı, eksik sütun/olay, bozuk dosya), desteklenmeyen harita, özellik sırası.
Yavaş (RUN_SLOW=1):
  - CS2CD gerileme (3 maç: anubis maskesi, dumanlı öldürmeler, round başı + çoklu öldürme): yeni hattın 15 özelliği
    data/processed/kill_features.parquet ile BİREBİR aynı (değer, satır sayısı, oyuncu, tick, NaN konumu; tolerans yok).
    Not: CS2CD'de .dem yok → bu test adaptörden SONRAKİ hattı doğrular. Adaptör için aşağıdaki tutarlılık testleri.
  - Determinizm: aynı girdi iki kez → aynı çıktı.
  - Güncel demolar (data/raw/current/*/*.dem): smoke test + adaptör tutarlılığı (skor tablosu, round sayısı,
    öldürmede iki taraf hayatta ve farklı takımda, konumlar harita sınırında).
    RUN_SLOW=1 python -m pytest tests/test_analyze_demo.py
"""
import json
import os
import shutil
import sys
from pathlib import Path

import polars as pl
import pytest
from polars.testing import assert_frame_equal

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
import analyze_demo as AD  # noqa: E402
import demo_adapter as A   # noqa: E402

SLOW = pytest.mark.skipif(not os.environ.get("RUN_SLOW"), reason="yavaş test; RUN_SLOW=1 ile çalıştır")
KILL_FEATURES = Path("data/processed/kill_features.parquet")
REGRESSION = {
    "anubis, maske devrede": "no_cheater_present/108",
    "dumanlı öldürmeler": "no_cheater_present/67",
    "round başı + aynı tick'te çoklu öldürme": "no_cheater_present/88",
}
DEMO_OUT = Path("data/processed/analysis_test")    # git dışı; demo çıktıları testler arasında yeniden kullanılır
ID_COLS = ["kill_id", "match", "player", "tick", "weapon", "weapon_class"]


# ---------------------------------------------------------------- yardımcılar
def fake_match(tmp: Path, tickrate: float = 64.0, drop_col: str | None = None, drop_event: str | None = None,
               drop_field: str | None = None, map_name: str = "de_dust2") -> tuple[Path, dict]:
    """CS2CD şemasında küçük sentetik maç (10 oyuncu, 2 takım, 300 tick)."""
    rows = []
    for t in range(1, 301):
        for i in range(10):
            rows.append(dict(tick=t, steamid=f"Player_{i + 1}", X=0.0, Y=0.0, Z=0.0, yaw=0.0, pitch=0.0, is_alive=True,
                             team_num=2 if i < 5 else 3, duck_amount=0.0, last_place_name="Mid",
                             approximate_spotted_by=[], game_time=t / tickrate, total_rounds_played=0,
                             kills_total=0, deaths_total=0, is_freeze_period=False))
    ticks = pl.DataFrame(rows)
    if drop_col:
        ticks = ticks.drop(drop_col)
    m = tmp / "fake" / "1"
    m.parent.mkdir(parents=True, exist_ok=True)
    ticks.write_parquet(m.with_suffix(".parquet"))
    ev = {"player_death": [dict(tick=200, attacker_steamid="Player_1", user_steamid="Player_6", weapon="ak47",
                                headshot=True, penetrated=0, thrusmoke=False)],
          "weapon_fire": [dict(tick=199, user_steamid="Player_1", weapon="weapon_ak47")],
          "player_hurt": [dict(tick=200, attacker_steamid="Player_1", user_steamid="Player_6")],
          "CSstats_info": [dict(map=map_name)]}
    if drop_event:
        ev.pop(drop_event)
    if drop_field:
        for r in ev["player_death"]:
            r.pop(drop_field)
    m.with_suffix(".json").write_text(json.dumps(ev), encoding="utf-8")
    return m, ev


def stored_features(match: str, feats: list[str]) -> pl.DataFrame:
    """Kayıtlı eğitim tablosu → yeni hattın biçimi (oyuncu Player_N; is_sniper/is_pistol baseline.load ile aynı)."""
    return (pl.read_parquet(KILL_FEATURES).filter(pl.col("match") == match)
              .with_columns(is_sniper=(pl.col("weapon_class") == "sniper").cast(pl.Int8),
                            is_pistol=(pl.col("weapon_class") == "tabanca").cast(pl.Int8),
                            player=pl.lit("Player_") + pl.col("player").str.split("_").list.last())
              .select("player", "tick", "weapon", "weapon_class", *feats))


def assert_same_features(new: pl.DataFrame, old: pl.DataFrame, feats: list[str], label: str):
    key = ["player", "tick", "weapon", "weapon_class", *feats]
    a = new.select(key).sort(key, nulls_last=True)
    b = old.select(key).sort(key, nulls_last=True)
    assert a.height == b.height > 0, f"{label}: satır sayısı {a.height} vs {b.height}"
    assert a.schema == b.schema, f"{label}: veri tipleri farklı"
    assert_frame_equal(a, b, check_exact=True)                       # tolerans yok
    for c in feats:
        assert (a[c].is_null() == b[c].is_null()).all(), f"{label}: {c} NaN konumları farklı"


# ---------------------------------------------------------------- hızlı testler
def test_feature_order_matches_frozen_model():
    feats = json.loads(Path("models/v1/features.json").read_text(encoding="utf-8"))["features"]
    assert AD.model_features() == feats and len(feats) == 15


def test_gate_passes_on_clean_fake(tmp_path):
    m, ev = fake_match(tmp_path)
    g, _ = AD.gate_checks(m, ev)
    assert g["tickrate"] == 64.0 and g["warnings"] == [] and g["n_players"] == 10


def test_wrong_tickrate_stops(tmp_path):
    m, ev = fake_match(tmp_path, tickrate=128.0)
    with pytest.raises(AD.AnalysisError, match="tick hızı"):
        AD.gate_checks(m, ev)


@pytest.mark.parametrize("col", ["X", "yaw", "is_alive", "team_num", "game_time"])
def test_missing_tick_column_stops(tmp_path, col):
    m, ev = fake_match(tmp_path, drop_col=col)
    with pytest.raises(AD.AnalysisError, match="zorunlu tick sütunları eksik"):
        AD.gate_checks(m, ev)


@pytest.mark.parametrize("event", ["player_death", "weapon_fire", "player_hurt"])
def test_missing_event_stops(tmp_path, event):
    m, ev = fake_match(tmp_path, drop_event=event)
    with pytest.raises(AD.AnalysisError, match="zorunlu olay"):
        AD.gate_checks(m, ev)


@pytest.mark.parametrize("field", ["penetrated", "thrusmoke", "headshot"])
def test_missing_event_field_stops(tmp_path, field):
    m, ev = fake_match(tmp_path, drop_field=field)
    with pytest.raises(AD.AnalysisError, match="zorunlu olay/alan eksik"):
        AD.gate_checks(m, ev)


def test_missing_context_columns_warns_not_silent(tmp_path):
    m, ev = fake_match(tmp_path, drop_col="kills_total")
    g, _ = AD.gate_checks(m, ev)
    assert any("bağlam sütunları yok" in w for w in g["warnings"]) and g["match_start_tick"] is None


def test_team_structure_warns(tmp_path):
    m, ev = fake_match(tmp_path)
    t = pl.read_parquet(m.with_suffix(".parquet")).filter(pl.col("steamid") != "Player_10")
    t.write_parquet(m.with_suffix(".parquet"))
    g, _ = AD.gate_checks(m, ev)
    assert any("takım başına" in w for w in g["warnings"]) and any("9 farklı oyuncu" in w for w in g["warnings"])


def test_bad_files_stop(tmp_path):
    with pytest.raises(AD.AnalysisError, match="dosya yok"):
        AD.validate_file(tmp_path / "yok.dem")
    txt = tmp_path / "a.txt"
    txt.write_text("x")
    with pytest.raises(AD.AnalysisError, match=".dem"):
        AD.validate_file(txt)
    bad = tmp_path / "bozuk.dem"
    bad.write_bytes(b"HL2DEMO\x00" + b"\x00" * 100)    # CS:GO başlığı / bozuk dosya
    with pytest.raises(AD.AnalysisError, match="PBDEMS2"):
        AD.validate_file(bad)


@pytest.mark.skipif(not (Path("data/raw/no_cheater_present/67.parquet").exists() and KILL_FEATURES.exists()),
                    reason="CS2CD maçı yok")
def test_unsupported_map_does_not_invent_visibility(tmp_path):
    """Harita .tri'siz → 5 görünürlük sütunu null, 'UYGUN DEĞİL'; nişan özellikleri kayıtlı değerlerle aynı."""
    raw = tmp_path / "raw" / "no_cheater_present"
    raw.mkdir(parents=True)
    shutil.copy("data/raw/no_cheater_present/67.parquet", raw / "67.parquet")
    ev = json.loads(Path("data/raw/no_cheater_present/67.json").read_text(encoding="utf-8"))
    ev["CSstats_info"][0]["map"] = "de_bilinmeyen_harita"
    (raw / "67.json").write_text(json.dumps(ev), encoding="utf-8")

    work = AD.analyze(cs2cd="no_cheater_present/67", raw_dir=tmp_path / "raw", out_root=tmp_path / "out")
    qc = json.loads((work / "qc.json").read_text(encoding="utf-8"))
    f = pl.read_parquet(work / "features.parquet")
    feats = AD.model_features()
    assert qc["model_v1_applicable"] is False and "desteklenmiyor" in qc["model_v1_reasons"][0]
    assert qc["visibility_computable"] is False
    vis = ["inv_angle_med", "inv_frac_lt10", "prefov_angle", "wallbang", "reaction_ms"]
    assert all(f[c].null_count() == f.height for c in vis)                 # uydurulmadı
    nonvis = [c for c in feats if c not in vis]
    assert_same_features(f, stored_features("no_cheater_present/67", feats), nonvis, "desteklenmeyen harita, nişan")
    assert "UYGUN DEĞİL" in (work / "qc.md").read_text(encoding="utf-8")


# ---------------------------------------------------------------- yavaş: CS2CD gerileme + determinizm
@SLOW
@pytest.mark.parametrize("case, match", list(REGRESSION.items()))
def test_cs2cd_regression_exact(tmp_path, case, match):
    work = AD.analyze(cs2cd=match, out_root=tmp_path)
    feats = AD.model_features()
    new = pl.read_parquet(work / "features.parquet")
    assert new.columns == ID_COLS + feats, "özellik sütun sırası models/v1/features.json ile aynı değil"
    old = stored_features(match, feats)
    assert_same_features(new, old, feats, case)
    # oyuncu eşleşmesi ve öldürme tick'leri ayrıca (çoklu öldürmede tekrar sayısı dahil)
    assert (new.group_by("player", "tick").len().sort("player", "tick")
            .equals(old.group_by("player", "tick").len().sort("player", "tick")))
    qc = json.loads((work / "qc.json").read_text(encoding="utf-8"))
    assert qc["model_v1_applicable"] and qc["kills_pre_match_aim"] == 0


def _all_outputs(work: Path) -> dict:
    out = {p.name: pl.read_parquet(p) for p in sorted(work.glob("*.parquet"))}
    out["qc.json"] = json.loads((work / "qc.json").read_text(encoding="utf-8"))
    md = json.loads((work / "metadata.json").read_text(encoding="utf-8"))
    md.pop("run"); md.pop("git_commit")                       # süre/bellek/commit doğal olarak değişir
    out["metadata.json"] = md
    out["qc.md"] = (work / "qc.md").read_text(encoding="utf-8")
    return out


@SLOW
def test_determinism_same_input_same_output(tmp_path):
    a = _all_outputs(AD.analyze(cs2cd="no_cheater_present/67", out_root=tmp_path / "a"))
    b = _all_outputs(AD.analyze(cs2cd="no_cheater_present/67", out_root=tmp_path / "b"))
    assert a.keys() == b.keys()
    for k in a:
        if isinstance(a[k], pl.DataFrame):
            assert_frame_equal(a[k], b[k], check_exact=True), k
        else:
            assert a[k] == b[k], k


# ---------------------------------------------------------------- yavaş: güncel demolar
DEMOS = A.demo_files()
_RUNS: dict = {}


def run_demo(source: str, path: Path) -> Path:
    """Demo başına bir kez analiz (resume: ara çıktılar yeniden kullanılır)."""
    if path not in _RUNS:
        _RUNS[path] = AD.analyze(path, source=source, out_root=DEMO_OUT, resume=True)
    return _RUNS[path]


def demo_param():
    return pytest.mark.parametrize("source, path", DEMOS, ids=[f"{s}-{i}" for i, (s, _) in enumerate(DEMOS)])


@SLOW
@demo_param()
def test_demo_smoke(source, path):
    work = run_demo(source, path)
    for f in ["metadata.json", "features.parquet", "kills.parquet", "kill_windows.parquet", "map_context.parquet",
              "utility.parquet", "qc.json", "qc.md"]:
        assert (work / f).exists(), f
    f = pl.read_parquet(work / "features.parquet")
    assert f.columns == ID_COLS + AD.model_features()
    qc = json.loads((work / "qc.json").read_text(encoding="utf-8"))
    assert qc["tickrate"] == 64.0 and qc["kills_model"] == f.height > 0
    assert qc["model_v1_applicable"] is True, qc["model_v1_reasons"]
    assert f["player"].str.starts_with("Player_").all()
    k = pl.read_parquet(work / "kills.parquet")
    assert set(f["kill_id"].to_list()) <= set(k.filter(pl.col("in_model"))["kill_id"].to_list())
    assert not k.filter(pl.col("in_model"))["pre_match"].any()          # ısınma/bıçak roundu modele girmez


@SLOW
@demo_param()
def test_adapter_vs_demo_scoreboard(source, path):
    """Maç başlangıcından sonraki öldürme/ölüm sayıları, demonun son tick'teki skor tablosuyla tutuyor mu?
    (CS2: takım öldürmesi kills_total'a eklenmez; intihar/dünya ölümü deaths_total'a eklenir)."""
    work = run_demo(source, path)
    did = work.name
    t = pl.read_parquet(AD.CACHE_ROOT / did / f"{did}.parquet", columns=["tick", "steamid", "team_num", "kills_total",
                                                                         "deaths_total", "team_rounds_total"])
    last = t.filter(pl.col("tick") == t["tick"].max())
    k = pl.read_parquet(work / "kills.parquet").filter(~pl.col("pre_match"))
    real = k.filter(pl.col("attacker").is_not_null() & (pl.col("attacker") != pl.col("victim"))
                    & (pl.col("attacker_team_num") != pl.col("victim_team_num")))
    kc = real.group_by("attacker").len().rename({"attacker": "steamid", "len": "ev_kills"})
    dc = k.group_by("victim").len().rename({"victim": "steamid", "len": "ev_deaths"})
    j = (last.select("steamid", "kills_total", "deaths_total").join(kc, on="steamid", how="left")
             .join(dc, on="steamid", how="left").fill_null(0))
    assert (j["kills_total"] == j["ev_kills"]).all(), j.filter(pl.col("kills_total") != pl.col("ev_kills"))
    assert (j["deaths_total"] == j["ev_deaths"]).all(), j.filter(pl.col("deaths_total") != pl.col("ev_deaths"))
    # Round sayısı = son tick'teki iki takımın skor toplamı
    qc = json.loads((work / "qc.json").read_text(encoding="utf-8"))
    score = last.group_by("team_num").agg(pl.col("team_rounds_total").max())["team_rounds_total"].sum()
    assert qc["rounds"] == score, (qc["rounds"], score)
    rounds = pl.read_parquet(work / "kills.parquet").filter(pl.col("in_model"))["round"]
    assert rounds.min() >= 1 and rounds.max() <= qc["rounds"]          # round'u bitiren öldürme kendi round'unda


@SLOW
@demo_param()
def test_adapter_kill_participants(source, path):
    """Modele giren her öldürmede saldırgan ve kurban bir önceki tick'te hayatta; takım öldürmeleri dışında
    farklı takımda. Takım öldürmesi sayısı QC'deki değerle aynı."""
    work = run_demo(source, path)
    k = pl.read_parquet(work / "kills.parquet").filter(pl.col("in_model"))
    assert k["attacker_alive_prev"].all() and k["victim_alive_prev"].all(), \
        k.filter(~pl.col("attacker_alive_prev") | ~pl.col("victim_alive_prev")).select("kill_id", "tick", "weapon")
    assert k["attacker_team_num"].is_in([2, 3]).all() and k["victim_team_num"].is_in([2, 3]).all()
    tk = k.filter(pl.col("attacker_team_num") == pl.col("victim_team_num")).height
    qc = json.loads((work / "qc.json").read_text(encoding="utf-8"))
    assert tk <= qc["teamkills"]


@SLOW
@demo_param()
def test_adapter_positions_within_map(source, path):
    work = run_demo(source, path)
    qc = json.loads((work / "qc.json").read_text(encoding="utf-8"))
    assert qc["positions_out_of_bounds"] == 0
    lo, hi = AD.map_bounds(qc["map"])
    m = AD.BOUNDS_MARGIN
    k = pl.read_parquet(work / "kills.parquet").filter(pl.col("in_model"))
    for role in ("attacker", "victim"):
        for i, ax in enumerate("XYZ"):
            c = k[f"{role}_{ax}"]
            assert c.null_count() == 0 and c.min() >= lo[i] - m and c.max() <= hi[i] + m, (role, ax)


@SLOW
@pytest.mark.skipif(not DEMOS, reason="güncel demo yok")
def test_adapter_deterministic_ids(tmp_path):
    """Aynı demo iki kez dönüştürülünce Player_N ataması ve bütün çıktılar aynı."""
    source, path = DEMOS[0]
    for d in ("a", "b"):
        A.convert(source, path, 0, out_dir=tmp_path / d, match_id="x")
    assert_frame_equal(pl.read_parquet(tmp_path / "a" / "x.parquet"), pl.read_parquet(tmp_path / "b" / "x.parquet"))
    assert (tmp_path / "a" / "x.json").read_text(encoding="utf-8") == (tmp_path / "b" / "x.json").read_text(encoding="utf-8")
    assert (tmp_path / "a" / "x.ids.json").read_text() == (tmp_path / "b" / "x.ids.json").read_text()


@SLOW
@demo_param()
def test_round_ending_kill_stays_in_its_round_and_live_paths_have_no_teleports(source, path):
    """total_rounds_played round'u bitiren öldürmenin tick'inde artar; bu öldürme kendi round'unda kalmalı.
    map_context 'live' aşamasında aynı round içinde ardışık örnekler arasında ışınlanma (büyük sıçrama) olmamalı."""
    work = run_demo(source, path)
    did = work.name
    t = (pl.read_parquet(AD.CACHE_ROOT / did / f"{did}.parquet", columns=["tick", "total_rounds_played"])
           .group_by("tick").agg(pl.col("total_rounds_played").max()).sort("tick"))
    inc = t.filter(pl.col("total_rounds_played").diff() > 0).select("tick", before=pl.col("total_rounds_played") - 1)
    k = pl.read_parquet(work / "kills.parquet")
    ending = k.join(inc, on="tick").filter(~pl.col("pre_match") & (pl.col("attacker") != pl.col("victim")))
    assert ending.height > 0 and (ending["round"] == ending["before"] + 1).all()
    assert set(k["phase"].unique().to_list()) <= {"pre_match", "freeze", "live", "post"}
    c = pl.read_parquet(work / "map_context.parquet").filter(pl.col("is_alive")).sort("player", "tick")
    j = c.with_columns(
        dist=((pl.col("X").diff() ** 2 + pl.col("Y").diff() ** 2).sqrt()).over("player"),
        cont=((pl.col("round").diff() == 0) & (pl.col("tick").diff() == AD.CONTEXT_STEP) & (pl.col("phase") == "live")
              & (pl.col("phase").shift(1) == "live")).over("player")).filter(pl.col("cont"))
    assert j["dist"].max() < 150            # 8 tick (125 ms) ~ en fazla ~40 birim koşu; 150 ışınlanmayı yakalar
