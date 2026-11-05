"""Faz 3.5 çok katlı haritalar — yalnız gösterim — ve duman/molotof eşleme düzeltmesi.

- politika: Nuke + Vertigo kat seçimi; Train tek kat (not); tek katlı haritalar etkilenmez
- ±32 birim histerezis; iz bölme; oyuncu katı geçmişle
- utility_intervals: entityid yeniden kullanımında SONRAKİ ilk bitiş; bitişsiz/uzun → ömürle sınırlı
- oynatma: kat görseli, diğer kattakiler 8–9 numaralı soluk izlerde, iz sayısı sabit
- arayüz (HLTV Nuke yerelde varsa): olay → saldırganın katı; elle seçim korunur; farklı kat notu; oynatmayla paylaşım
"""
import json
import sys
from pathlib import Path

import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
import playback as PB  # noqa: E402
import radar as R  # noqa: E402
import viewer_data as V  # noqa: E402

APP = str(Path(__file__).parent.parent / "app" / "app.py")
NUKE = next((e for e in V.list_analyses() if e["map"] == "de_nuke"), None)
INFO = dict(multi_level=True, levels=[dict(name="default", image="u.png", z_min=-495.0, z_max=1e4),
                                      dict(name="lower", image="l.png", z_min=-1e4, z_max=-495.0)])


def test_policy_train_single_level_and_others_unaffected():
    assert V.MULTILEVEL_MAPS == {"de_nuke", "de_vertigo"}
    assert V.level_enabled(INFO, "de_nuke") and V.level_enabled(INFO, "de_vertigo")
    assert not V.level_enabled(INFO, "de_train")                           # Train: tek kat
    assert V.SINGLE_LEVEL_NOTE["de_train"] == ("Train için mevcut radar verisi katları güvenilir biçimde ayırmıyor; "
                                               "tek kat görünümü kullanılıyor.")
    assert not V.level_enabled(dict(INFO, multi_level=False), "de_nuke")
    assert not V.level_enabled(None, "de_nuke")


def test_hysteresis_32_units():
    b = -495.0
    assert V.LEVEL_HYST == 32.0
    # sınırın hemen altı/üstü kat değiştirmez; 32 birimden fazla geçince değişir
    assert V.assign_levels([-400, -500, -520, -480, -540, -400, -470, -450], b) == \
        ["default", "default", "default", "default", "lower", "default", "default", "default"]
    assert V.assign_levels([-600, -480, -470, -462], b) == ["lower", "lower", "lower", "default"]
    assert V.assign_levels([], b) == []


def test_trail_segments_split_by_level():
    g = pl.DataFrame({"X": [0.0, 1, 2, 3, 4], "Y": [0.0, 0, 0, 0, 0], "Z": [-400.0, -400, -600, -600, -600]})
    (sx, sy), (ox, oy) = V.trail_segments(g, INFO, "default")
    assert [x for x in sx if x is not None] == [0, 1, 2]                   # geçiş noktası iki tarafta da (kopma yok)
    assert [x for x in ox if x is not None] == [1, 2, 3, 4] and None in sx   # seçili kat parçası geçişte kesilir


def test_player_level_uses_round_history():
    ctx = pl.DataFrame({"player": ["A"] * 4, "tick": [8, 16, 24, 32], "Z": [-600.0, -500, -480, -470]})
    assert V.player_level(ctx, INFO, "A", 32) == "lower"                   # -470: sınırı 32'den fazla aşmadı
    assert V.player_level(ctx, INFO, "A", 32, z_now=-400) == "default"


def test_utility_intervals_reused_entity_and_cap():
    u = pl.DataFrame({
        "tick": [100, 1500, 5000, 9000, 20000],
        "entityid": [7, 7, 7, 7, 9],
        "x": [1.0, 0, 2, 0, 3], "y": [1.0, 0, 2, 0, 3], "z": [0.0, 0, 0, 0, 0], "player": ["p"] * 5,
        "event": ["smokegrenade_detonate", "smokegrenade_expired", "smokegrenade_detonate", "smokegrenade_expired",
                  "inferno_startburn"]})
    iv = V.utility_intervals(u).sort("tick")
    assert iv.height == 3                                                 # başlangıç başına tek aralık
    assert iv["end_tick"].to_list() == [1500, 5000 + int(25 * 64), 20000 + int(8 * 64)]
    # 2. duman: sonraki bitiş 9000 ama ömür üst sınırı 25 sn; molotof bitişsiz → 8 sn
    assert V.active_utility(u, 1499).height == 1 and V.active_utility(u, 1500).height == 0
    assert V.active_utility(u, 3000).height == 0                          # eski hata: 7 numaralı duman burada "etkin"di


def test_utility_intervals_real_analyses_one_per_start():
    for e in V.list_analyses():
        u = pl.read_parquet(Path(e["work"]) / "utility.parquet")
        iv = V.utility_intervals(u)
        starts = u.filter(pl.col("event").is_in(["smokegrenade_detonate", "inferno_startburn"])).height
        assert iv.height == starts, e["demo_id"]
        assert ((iv["end_tick"] - iv["tick"]) <= 25 * 64).all() and ((iv["end_tick"] - iv["tick"]) >= 0).all()


@pytest.mark.skipif(NUKE is None, reason="yerelde Nuke analizi yok")
def test_playback_levels_traces():
    w = Path(NUKE["work"])
    rnd = 3
    f0, t0 = PB.build_playback_figure(w, rnd)
    fu, tu = PB.build_playback_figure(w, rnd, level="default")
    fl, tl = PB.build_playback_figure(w, rnd, level="lower")
    assert t0["level"] is None and tu["level"] == "default" and tl["level"] == "lower"
    assert len(f0["frames"]) == len(fu["frames"]) == len(fl["frames"])
    assert all(len(fr["data"]) == 11 for fr in fu["frames"])               # iz sayısı sabit (animasyon bozulmaz)
    assert fu["layout"]["images"][0]["source"] != fl["layout"]["images"][0]["source"]   # kat görseli
    tot = lambda f, ks: sum(len(fr["data"][k]["x"]) for fr in f["frames"] for k in ks)  # noqa: E731
    assert tot(f0, (8, 9)) == 0                                            # kat kapalı: diğer kat izi boş
    assert tot(fu, (2, 3)) + tot(fu, (8, 9)) == tot(f0, (2, 3))            # her oyuncu tam bir yerde
    assert tot(fu, (8, 9)) == tot(fl, (2, 3)) and tot(fu, (8, 9)) > 0
    hv = [h for fr in fu["frames"] for k in (8, 9) for h in fr["data"][k]["hovertext"]]
    assert hv and all(h.endswith("(alt kat)") for h in hv)


def _pick(at, demo, ev, kid, nonce):
    at.session_state[f"tl_pick_{demo}"] = V.timeline_pick([{"customdata": kid}], ev)
    at.session_state[f"tl_nonce_{demo}"] = nonce
    return at.run()


@pytest.mark.skipif(NUKE is None, reason="yerelde Nuke analizi yok")
def test_app_auto_and_manual_level_shared_with_playback():
    from streamlit.testing.v1 import AppTest
    d = V.load(NUKE)
    demo, ev, info = d["meta"]["demo_id"], d["events"], d["radar"]
    lv = {}
    for r in ev.iter_rows(named=True):
        lv[r["kill_id"]] = V.event_levels(V.context_round(d["work"], r["round"]), info, r)
    low = next(k for k, (a, v) in lv.items() if a == "lower" and v == "lower")
    up = next(k for k, (a, v) in lv.items() if a == "default" and v == "default")
    cross = next(k for k, (a, v) in lv.items() if a != v)
    at = AppTest.from_file(APP, default_timeout=180)
    at.query_params["demo"] = demo
    at.run()
    key = f"level_{demo}"
    _pick(at, demo, ev, low, 1)
    assert not at.exception and at.session_state[key] == "Alt kat"          # olay → saldırganın katı
    at.session_state[key] = "Üst kat"                                      # elle değişim
    at.run()
    assert at.session_state[key] == "Üst kat"
    assert any("şu an üst kat gösteriliyor" in x.value for x in at.markdown)
    at.session_state[f"mode_{demo}"] = "Round Oynat"                       # oynatma aynı seçimi kullanır
    at.run()
    assert not at.exception and at.session_state[key] == "Üst kat"
    pb = next(json.loads(c.proto.spec) for c in at.get("plotly_chart") if json.loads(c.proto.spec).get("frames"))
    assert len(pb["frames"][0]["data"]) == 11
    _pick(at, demo, ev, up, 2)                                             # başka olay → otomatik katına döner
    assert at.session_state[key] == "Üst kat" and at.session_state[f"mode_{demo}"] == "Olay İncele"
    _pick(at, demo, ev, low, 3)
    assert at.session_state[key] == "Alt kat"
    _pick(at, demo, ev, cross, 4)
    assert any("farklı katlar" in x.value for x in at.markdown)
    radar = json.loads(at.get("plotly_chart")[0].proto.spec)
    assert any("open" in str((t.get("marker") or {}).get("symbol", "")) for t in radar["data"])   # içi boş işaret
