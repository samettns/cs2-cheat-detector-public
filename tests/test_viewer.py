"""Faz 3 testleri: arayüz veri katmanı (src/viewer_data.py), CS2CD OOF gösterim modu (src/oof_display.py),
örnek seçim kuralı (src/jury_set.py) ve Streamlit arayüzünün hatasız açılması (app/app.py)."""
import json
import os
import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
import viewer_data as V  # noqa: E402

ANALYSIS = Path("reports/analysis")
JURY = Path("data/reference/v1/jury_set.json")


# ---------------------------------------------------------------- veri katmanı
def test_aim_endpoint_yaw_convention():
    x, y = V.aim_endpoint(0, 0, 0, 100)
    assert (round(x), round(y)) == (100, 0)            # yaw 0 → +X (doğu)
    x, y = V.aim_endpoint(0, 0, 90, 100)
    assert (round(x), round(y)) == (0, 100)            # yaw 90 → +Y (kuzey; radarda yukarı)
    x, y = V.aim_endpoint(10, 10, -180, 50)
    assert (round(x), round(y)) == (-40, 10)


def _ctx():
    rows = []
    for t in range(0, 80, 8):
        phase = "freeze" if t < 16 else ("live" if t < 64 else "post")
        for p, team in (("Player_1", 2), ("Player_2", 3)):
            rows.append(dict(tick=t, player=p, X=float(t), Y=0.0, Z=0.0, yaw=0.0, pitch=0.0, is_alive=not (p == "Player_2" and t >= 48),
                             team_num=team, round=3, phase=phase))
    return pl.DataFrame(rows)


def test_live_trails_only_live_and_alive():
    tr = V.live_trails(_ctx(), 72)
    assert set(tr) == {"Player_1", "Player_2"}
    assert tr["Player_1"]["tick"].to_list() == [16, 24, 32, 40, 48, 56]       # freeze ve post yok
    assert tr["Player_2"]["tick"].max() == 40                                # öldükten sonra yok


def test_positions_at_uses_latest_sample_not_after():
    c = _ctx()
    assert V.positions_at(c, 30)["tick"].unique().to_list() == [24]
    assert V.positions_at(c, 0)["tick"].unique().to_list() == [0]


def test_active_utility_window():
    u = pl.DataFrame(dict(tick=[100, 400, 150, 300], entityid=[1, 1, 2, 2], x=[0.0, 0.0, 5.0, 5.0], y=[0.0] * 4,
                          z=[0.0] * 4, player=["Player_1"] * 4,
                          event=["smokegrenade_detonate", "smokegrenade_expired", "inferno_startburn", "inferno_expire"]))
    assert V.active_utility(u, 99).height == 0
    a = V.active_utility(u, 200)
    assert sorted(a["kind"].to_list()) == ["molotof", "smoke"]
    assert V.active_utility(u, 350)["kind"].to_list() == ["smoke"]
    assert V.active_utility(u, 400).height == 0


# ---------------------------------------------------------------- örnek seçim kuralı
@pytest.mark.skipif(not JURY.exists(), reason="örnek seti yok")
@pytest.mark.skipif(not Path("data/maps/radar/de_dust2.json").exists(),
                    reason="seçim kuralı yerel harita dosyalarına bağlı (data/maps/, git dışı); kayıtlı set donuk")
def test_jury_set_is_reproducible_and_follows_rule():
    import jury_set as J
    saved = json.loads(JURY.read_text(encoding="utf-8"))
    assert J.select() == saved                                       # aynı kural + seed → aynı seçim
    roles = [e["role"] for e in saved["examples"]]
    assert roles == ["dogrulanmis_pozitif", "hilecisiz", "yanlis_alarm"] and saved["seed"] == 2026
    maps = [e["map"] for e in saved["examples"][:2]]
    assert maps[0] != maps[1]
    a, b, c = saved["examples"]
    assert a["match"].startswith("with_cheater_present/") and b["match"].startswith("no_cheater_present/")
    assert c["match"].startswith("no_cheater_present/") and c["match"] not in J.D13_EXCLUDED
    assert "sniper" in c["note"]


# ---------------------------------------------------------------- OOF gösterim modu (örnek maçları üretildiyse)
JURY_WORKS = [ANALYSIS / ("cs2cd_" + e["match"].replace("/", "_"))
              for e in (json.loads(JURY.read_text(encoding="utf-8"))["examples"] if JURY.exists() else [])]
HAVE_OOF = [w for w in JURY_WORKS if (w / "oof_result.json").exists()]


@pytest.mark.skipif(not HAVE_OOF, reason="OOF gösterim çıktısı yok (python src/oof_display.py jury)")
@pytest.mark.parametrize("work", HAVE_OOF, ids=[w.name for w in HAVE_OOF])
def test_oof_display_matches_evaluation_and_labels_separate(work):
    import oof_display as O
    r = json.loads((work / "oof_result.json").read_text(encoding="utf-8"))
    match = r["match"]
    p = pl.read_parquet(work / "oof_player_scores.parquet")
    lab = pl.read_parquet(work / "dataset_labels.parquet")
    # oyuncu skoru = OOF öldürme skorlarının ortalaması (>= 3)
    k = pl.read_parquet(O.OOF_FILE).filter(pl.col("match") == match)
    m = k.group_by("player").agg(s=pl.col("score").mean(), n=pl.len()).filter(pl.col("n") >= 3)
    j = p.drop_nulls("player_raw_score").join(m, on="player")
    assert j.height == m.height and np.array_equal(j["player_raw_score"].to_numpy(), j["s"].to_numpy())
    # hilecisiz maç: OOF oyuncu skorları dondurulmuş clean_reference ile birebir
    if match.startswith("no_cheater_present/"):
        ref = pl.read_parquet("models/v1/clean_reference.parquet").with_columns(
            player_short=pl.lit("Player_") + pl.col("player").str.split("_").list.last(),
            match=pl.col("player").str.replace(r"_Player_\d+$", ""))
        ref = ref.filter(pl.col("match") == match).select(player="player_short", score="score")
        jj = p.drop_nulls("player_raw_score").join(ref, on="player")
        assert jj.height == ref.height > 0 and np.array_equal(jj["player_raw_score"].to_numpy(), jj["score"].to_numpy())
        assert (lab["dataset_label"] == O.LABEL_CLEAN).all()
    else:
        ch = set(k.filter(pl.col("y") == 1)["player"].to_list())
        for row in lab.iter_rows(named=True):
            assert row["dataset_label"] == (O.LABEL_CHEATER if row["player"] in ch else O.LABEL_UNCERTAIN)
        # belirsiz etiketliler değerlendirme dışı: skor yok
        und = p.filter(pl.col("review_priority") == O.NOT_EVALUATED)
        assert und["player_raw_score"].is_null().all()
    # model çıktısı etiketi içermez
    assert "dataset_label" not in p.columns
    assert r["mode_note"] == "Eğitim veri havuzundan — Out-of-Fold / çapraz doğrulama skoru"


# ---------------------------------------------------------------- arayüz biçimlendirme yardımcıları
def test_player_order_priority_then_percentile():
    p = pl.DataFrame(dict(player=["Player_1", "Player_2", "Player_3", "Player_4", "Player_5", "Player_10"],
                          review_priority=["düşük inceleme önceliği", "yüksek inceleme önceliği", "düşük inceleme önceliği",
                                           "yetersiz veri", "orta inceleme önceliği", "düşük inceleme önceliği"],
                          style_percentile=[40.0, 96.0, 80.0, None, 91.0, 80.0]))
    assert V.player_order(p) == ["Player_2", "Player_5", "Player_3", "Player_10", "Player_1", "Player_4"]


def test_missing_values_shown_as_unknown_and_flags_as_words():
    assert V.fmt_value(None) == V.fmt_value(float("nan")) == "bilinmiyor"
    assert V.fmt_value(117.2, 0, " ms") == "117 ms"
    assert (V.yes_no(True), V.yes_no(False), V.yes_no(None)) == ("evet", "hayır", "bilinmiyor")
    assert V.visibility_text(None) == "bilinmiyor" and V.visibility_text(False) == "görünmez"


def test_explanation_bullets_at_most_three():
    tf = json.dumps([dict(text=f"özellik {i} 1.0 — temiz oyuncuların %9{i}'sinden yüksek") for i in range(5)])
    b = V.explanation_bullets(tf)
    assert len(b) == 3 and b[0] == ("özellik 0 1.0", "temiz oyuncuların %90'sinden yüksek")
    assert len(V.explanation_bullets(None, "a — x; b — y; c — z; d — w")) == 3
    assert V.explanation_bullets(None, None) == []


# ---------------------------------------------------------------- arayüz (Streamlit AppTest)
APP = str(Path(__file__).parent.parent / "app" / "app.py")


def _texts(at) -> str:
    parts = [x.value for x in at.markdown] + [x.value for x in at.caption] + [x.value for x in at.info]
    parts += [x.value for x in at.warning] + [x.value for x in at.error]
    return " ".join(str(x) for x in parts)


@pytest.mark.skipif(not V.list_analyses(), reason="arayüzde açılabilecek analiz yok")
def test_app_runs_for_every_analysis_without_errors():
    from streamlit.testing.v1 import AppTest
    entries = V.list_analyses()
    for i, e in enumerate(entries):
        at = AppTest.from_file(APP, default_timeout=180).run()
        if i:
            at.sidebar.selectbox[0].set_value(i).run()
        assert not at.exception, (e["title"], [x.value for x in at.exception])
        text = _texts(at)
        assert "kalibre edilmiş hile olasılığı değildir" in text            # kısa uyarı her zaman görünür
        assert "Olay zaman çizelgesi" in text
        assert any(x.label == "Model sınırlılıkları" for x in at.expander)
        assert any(x.label == "Detaylı veriler" for x in at.expander)
        assert "nan" not in text.split() and "None" not in text.split()     # eksik değer = 'bilinmiyor'
        if e["mode"] == "cs2cd_oof":
            assert "Out-of-Fold" in text
            assert any("model çıktısı DEĞİLDİR" in x.label for x in at.expander)
        else:
            assert not any("Veri seti etiketi" in x.label for x in at.expander)   # güncel demoda etiket yok
        if e.get("jury"):
            assert "Bu örnek neyi gösteriyor?" in text
            if e["jury"]["role"] == "yanlis_alarm":
                assert "sniper ağırlıklı oyuncularda yanlış alarm riski" in text
        # model çıktısı alanlarında (oyuncu kartları, ayrıntı kutuları) kesin hüküm yok
        cards = " ".join(x.value for x in at.markdown if 'class="pcard' in x.value or 'class="kv"' in x.value)
        assert cards and "hileci" not in cards.lower()


@pytest.mark.skipif(not V.list_analyses(), reason="arayüzde açılabilecek analiz yok")
def test_app_player_cards_sorted_and_selectable():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(APP, default_timeout=180).run()
    E = V.list_analyses()
    d = V.load(E[V.default_index(E)])          # açılış demosu
    order = V.player_order(d["players"])
    cards = [x.value for x in at.markdown if 'class="pcard' in x.value]
    names = [next(p for p in order if f">{p}<" in c) for c in cards]
    assert names == order
    btn = next(b for b in at.button if b.key.startswith("pick_"))
    target = btn.key.removeprefix("pick_")
    btn.click().run()
    assert not at.exception
    assert any('class="pcard sel"' in x.value and f">{target}<" in x.value for x in at.markdown)


# ---------------------------------------------------------------- UI düzeltmeleri
def test_live_rounds_drop_post_match_half_round():
    meta = dict(rounds=2, rounds_table=[dict(round=1, start_tick=0, end_tick=10, live_start_tick=2),
                                        dict(round=2, start_tick=11, end_tick=20, live_start_tick=13),
                                        dict(round=3, start_tick=21, end_tick=25, live_start_tick=None)])
    assert [r["round"] for r in V.live_rounds(meta)] == [1, 2]


@pytest.mark.skipif(not V.list_analyses(), reason="analiz yok")
def test_live_rounds_match_reported_round_count():
    for e in V.list_analyses():
        d = V.load(e)
        rt = V.live_rounds(d["meta"])
        assert len(rt) == d["meta"]["rounds"] and rt[-1]["round"] == d["meta"]["rounds"], e["demo_id"]
        assert d["events"]["round"].max() <= d["meta"]["rounds"]


def test_metric_cards_short_and_grouped():
    tf = json.dumps([dict(feature="reaction_ms", value=117.2, notable_pct=95.7, direction=-1,
                          text="tepki süresi 117 ms — temiz sniper ağırlıklı oyuncuların oyuncu ortalamalarının %96'sından kısa"),
                     dict(feature="prefov_angle", value=3.6, notable_pct=60.0, direction=-1,
                          text="x — temiz ... %60'ından küçük (tipik aralıkta)"),
                     dict(feature="jitter", value=0.61, notable_pct=91.0, direction=1, text="y — %91'inden yüksek"),
                     dict(feature="shots", value=1, notable_pct=90.0, direction=-1, text="z")])
    c = V.metric_cards(tf)
    assert len(c) == 3
    assert (c[0]["group"], c[0]["name"], c[0]["value"], c[0]["compare"]) == ("Tepki", "Tepki süresi", "117 ms",
                                                                          "temizlerin %96'sından kısa")
    assert c[1]["group"] == "Görünürlük / pre-aim" and c[1]["typical"] and c[2]["group"] == "Nişan davranışı"


@pytest.mark.skipif(not V.list_analyses(), reason="analiz yok")
def test_review_point_consistent_with_event_threshold():
    """İnceleme noktası = olay skoru > stil içi olay eşiği; oyuncu önceliği ayrı (ortalama) eşik."""
    for e in V.list_analyses():
        ev = V.load(e)["events"]
        assert (ev["review_point"] == (ev["event_score"] > ev["event_threshold"])).all()


def _open(demo_id):
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(APP, default_timeout=180)
    at.query_params["demo"] = demo_id
    return at.run()


@pytest.mark.skipif(not any(e.get("jury") and e["jury"].get("flagged_players") for e in V.list_analyses()),
                    reason="bakılacak oyuncusu olan örnek örneği yok")
def test_jury_flagged_player_selected_on_open():
    for e in V.list_analyses():
        fp = (e.get("jury") or {}).get("flagged_players")
        if not fp:
            continue
        at = _open(e["demo_id"])
        assert not at.exception
        assert any('class="pcard sel"' in x.value and f">{fp[0]}<" in x.value for x in at.markdown)


@pytest.mark.skipif(not V.list_analyses(), reason="analiz yok")
def test_app_shows_event_threshold_kill_counts_and_new_terms():
    e = V.list_analyses()[0]
    at = _open(e["demo_id"])
    text = _texts(at)
    assert "Olay eşiği (stil içi %95)" in text and "Olay ayrıntıları" in [x.label for x in at.expander]
    assert "İnceleme noktası nedir?" in text and V.POINT_DEF in " ".join(x.value for x in at.caption)   # taşındı
    assert "Model skoru" in text and "Ham risk skoru" not in text
    d = V.load(e)
    cards = [x.value for x in at.markdown if 'class="pcard' in x.value]
    for p, k in d["players"].select("player", "analyzed_kills").iter_rows():
        c = next(c for c in cards if f">{p}<" in c)
        assert f"<b>{k}</b> öldürme" in c
        assert ("az veri" in c) == (k < V.LOW_DATA_KILLS)


@pytest.mark.skipif(not V.list_analyses(), reason="analiz yok")
def test_upload_form_defaults_keep_raw_off_and_start_disabled():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(APP, default_timeout=180).run()
    assert not at.exception
    keep = next(c for c in at.checkbox if c.key == "dem_keep")
    assert keep.value is False and "sakla" in keep.label
    start = next(b for b in at.button if b.key == "dem_start")
    assert start.disabled                                  # dosya seçilmeden başlatılamaz
    assert at.sidebar.selectbox[0].label == "Önceden analiz edilmiş demolar"


def test_default_demo_is_not_hltv():
    """?demo= yoksa açılış demosu HLTV (profesyonel) olmaz; ?demo= verilirse o açılır."""
    E = [dict(demo_id="a1", source="hltv"), dict(demo_id="b2", source="faceit"), dict(demo_id="c3", source="valve")]
    assert V.default_index(E) == 1
    assert V.default_index(E, "a1") == 0
    assert V.default_index(E, "yok") == 1
    assert V.default_index([dict(demo_id="a1", source="hltv")]) == 0
    assert V.default_index([dict(demo_id="x", source=None, training=True)]) == 0
    real = V.list_analyses()
    if real and any(e.get("source") not in V.NOT_DEFAULT_SOURCES for e in real):
        assert real[V.default_index(real)].get("source") != "hltv"


# ---------------------------------------------------------------- Faz 3.5: zaman çizelgesinden olay seçimi
def _ev():
    return pl.DataFrame({"kill_id": [5, 9, 12], "attacker": ["Player_1", "Player_2", "Player_1"],
                         "round": [1, 2, 3], "kill_tick": [100, 200, 300]})


def test_timeline_pick_uses_kill_id_not_index():
    ev = _ev()
    # point_index / curve_number bilerek yanlış: seçim yalnız customdata (kill_id) ile yapılmalı
    p = V.timeline_pick([{"customdata": 9, "point_index": 0, "curve_number": 1}], ev)
    assert p == dict(kill_id=9, player="Player_2", round=2, tick=200)
    assert V.timeline_pick([{"customdata": [12], "point_index": 99}], ev)["player"] == "Player_1"   # liste biçimi
    # filtre sonrası sıra değişse de (olaylar ters sırada) aynı olay
    assert V.timeline_pick([{"customdata": 9}], ev.reverse()) == p


def test_timeline_pick_empty_or_unknown():
    ev = _ev()
    assert V.timeline_pick(None, ev) is None
    assert V.timeline_pick([], ev) is None
    assert V.timeline_pick([{"point_index": 1}], ev) is None            # customdata yok (ör. round bandı)
    assert V.timeline_pick([{"customdata": 777}], ev) is None           # bilinmeyen kill_id
    assert V.timeline_pick([{"customdata": None}, {"customdata": 5}], ev)["kill_id"] == 5


def _app_with_pick(at, demo, pick, nonce):
    at.session_state[f"tl_pick_{demo}"] = pick
    at.session_state[f"tl_nonce_{demo}"] = nonce
    return at.run()


@pytest.mark.skipif(not V.list_analyses(), reason="analiz yok")
def test_timeline_click_selects_attacker_event_round_tick():
    from streamlit.testing.v1 import AppTest
    E = V.list_analyses()
    d = V.load(E[V.default_index(E)])
    demo = d["meta"]["demo_id"]
    at = AppTest.from_file(APP, default_timeout=180).run()
    assert not at.exception
    current = at.session_state[f"player_{demo}"]
    # başka bir oyuncunun, o oyuncunun en yüksek skorlu OLMAYAN bir olayı (varsayılan seçimle karışmasın)
    other = d["events"].filter(pl.col("attacker") != current)
    att = other.group_by("attacker").len().filter(pl.col("len") >= 2)["attacker"][0]
    row = other.filter(pl.col("attacker") == att).sort("event_score").row(0, named=True)
    pick = V.timeline_pick([{"customdata": row["kill_id"]}], d["events"])
    _app_with_pick(at, demo, pick, 1)
    assert not at.exception
    assert at.session_state[f"player_{demo}"] == att                                     # saldırgan seçili oyuncu
    assert any('class="pcard sel flash"' in x.value and f">{att}<" in x.value for x in at.markdown)   # görsel vurgu
    assert any("Oyuncu değişti" in t.value for t in at.toast)
    assert any("Zaman çizelgesinden seçildi" in x.value for x in at.markdown)
    assert at.selectbox(key=f"ev_{demo}_{att}").value == row["kill_id"]                 # olay seçili
    rnd = next(s for s in at.selectbox if s.key.startswith(f"rnd_{demo}_{att}_{row['kill_id']}_"))
    assert rnd.value == row["round"]                                                     # round
    tick = next(s for s in at.slider if s.key.startswith(f"tick_{demo}_{att}_{row['kill_id']}_"))
    assert tick.value == row["kill_tick"]                                                # tick = olay anı (radar oraya)

    # kullanıcı tick'i kaydırır, sonra AYNI noktaya tekrar tıklar → tick yine olay anına döner
    tick.set_value(tick.min).run()
    assert next(s for s in at.slider if s.key.startswith(f"tick_{demo}_{att}_")).value == tick.min
    _app_with_pick(at, demo, pick, 2)
    tick2 = next(s for s in at.slider if s.key.startswith(f"tick_{demo}_{att}_{row['kill_id']}_"))
    assert tick2.value == row["kill_tick"] and not at.exception
    assert not any("pcard sel flash" in x.value for x in at.markdown)                   # oyuncu değişmedi → vurgu yok

    # mevcut yollar: dropdown ile başka olay → round/tick o olaya; kart butonu ile oyuncu değişimi
    evs = d["events"].filter(pl.col("attacker") == att)
    other_kid = evs.filter(pl.col("kill_id") != row["kill_id"])["kill_id"][0]
    at.selectbox(key=f"ev_{demo}_{att}").set_value(other_kid).run()
    o = evs.filter(pl.col("kill_id") == other_kid).row(0, named=True)
    assert next(s for s in at.slider if s.key.startswith(f"tick_{demo}_{att}_{other_kid}_")).value == o["kill_tick"]
    btn = next(b for b in at.button if b.key.startswith("pick_"))
    target = btn.key.removeprefix("pick_")
    btn.click().run()
    assert not at.exception and at.session_state[f"player_{demo}"] == target


# ---------------------------------------------------------------- Faz 3.5: olay filtreleri
def _fev():
    return pl.DataFrame({
        "kill_id": [1, 2, 3, 4, 5, 6], "attacker": ["A", "A", "B", "B", "C", "C"],
        "weapon_class": ["otomatik", "sniper", "otomatik", "tabanca", "otomatik", "otomatik"],
        "review_point": [True, False, True, None, False, True],
        "wallbang": [1.0, 0.0, 0.0, None, 1.0, 1.0],
        "thrusmoke": [False, True, False, None, True, False],
        "victim_visible_at_kill": [False, True, None, False, False, True],
        "inv_angle_med": [3.0, 1.0, 9.0, None, 4.0, 2.0],
        "event_style_percentile": [96.0, 50.0, 99.5, None, 80.0, 91.0]})


def test_filter_events_single_and_combined_and_nulls():
    ev, thr = _fev(), {"otomatik": 4.5, "sniper": 2.0, "tabanca": 3.5}
    ids = lambda a, p=0: V.filter_events(ev, a, p, thr)["kill_id"].to_list()   # noqa: E731
    assert ids([]) == [1, 2, 3, 4, 5, 6]
    assert ids(["review_point"]) == [1, 3, 6]                 # None eşleşmez
    assert ids(["wallbang"]) == [1, 5, 6]
    assert ids(["smoke"]) == [2, 5]
    assert ids(["not_visible"]) == [1, 4, 5]                  # bilinmiyor (None) eşleşmez
    assert ids(["aim_near_hidden"]) == [1, 2, 5, 6]           # silah sınıfı eşiği; NaN eşleşmez
    assert ids([], 90) == [1, 3, 6]
    assert ids(["review_point", "wallbang"]) == [1, 6]        # VE
    assert ids(["review_point", "wallbang", "not_visible"]) == [1]
    assert ids(list(V.FILTERS), 99) == []                     # boş sonuç
    assert V.filter_events(ev.reverse(), ["wallbang"], 0, thr)["kill_id"].to_list() == [6, 5, 1]   # sıra korunur
    assert V.filter_summary(["wallbang", "review_point"], 90, 2, 6) == \
        "Filtre: İnceleme noktası + Wallbang + stil içi olay yüzdeliği ≥ %90 · 2 / 6 olay"
    assert V.filter_summary([], 0, 6, 6) == "Filtre yok · 6 / 6 olay"


def test_filter_texts_are_cautious():
    assert "wallhack anlamına gelmez" in V.FILTERS["not_visible"][1]
    t = V.FILTERS["aim_near_hidden"][1]
    assert "tek başına hile göstergesi değildir" in t and "Model eşiği değildir" in t
    assert V.FILTERS["not_visible"][0] == "Öldürme anında görünür değil"
    assert V.FILTERS["aim_near_hidden"][0] == "Görünmez hedefe nişan yakınlığı yüksek"


def test_aim_near_thresholds_from_frozen_reference():
    thr = V.aim_near_thresholds()
    ref = pl.read_parquet("data/reference/v1/clean_kill_features.parquet").drop_nulls("inv_angle_med")
    for wc, v in thr.items():
        share = (ref.filter(pl.col("weapon_class") == wc)["inv_angle_med"] <= v).mean()
        assert 0.09 <= share <= 0.11                          # tanım: temizlerin en küçük %10'u


def _timeline_ids(at):
    """Zaman çizelgesi grafiğindeki noktaların kill_id'leri ('seçili olay' halkası hariç)."""
    import base64
    charts = at.get("plotly_chart")
    if not charts:
        return None
    fig = json.loads(charts[-1].proto.spec)
    out = []
    for tr in fig["data"]:
        if tr.get("name") == "seçili olay" or "customdata" not in tr:
            continue
        cd = tr["customdata"]
        if isinstance(cd, dict):                              # plotly ikili kodlama
            cd = np.frombuffer(base64.b64decode(cd["bdata"]), dtype=np.dtype(cd["dtype"])).tolist()
        out += [int(x) for x in cd]
    return sorted(out)


@pytest.mark.skipif(not V.list_analyses(), reason="analiz yok")
def test_app_filters_same_set_for_timeline_list_and_scores_unchanged():
    from streamlit.testing.v1 import AppTest
    E = V.list_analyses()
    d = V.load(E[V.default_index(E)])
    demo, ev = d["meta"]["demo_id"], d["events"]
    at = AppTest.from_file(APP, default_timeout=180).run()
    cards_before = [x.value for x in at.markdown if 'class="pcard' in x.value]
    assert _timeline_ids(at) == sorted(ev["kill_id"].to_list())          # filtre yok → bütün olaylar

    active = ["review_point"]
    fe = V.filter_events(ev, active, 0, V.aim_near_thresholds())
    at.session_state[f"flt_{demo}"] = active
    at.run()
    assert not at.exception
    assert _timeline_ids(at) == sorted(fe["kill_id"].to_list())          # zaman çizelgesi = filtreli küme
    player = at.session_state[f"player_{demo}"]
    sb = at.selectbox(key=f"ev_{demo}_{player}")
    import re
    opts = [int(re.search(r"#(\d+)", o).group(1)) for o in sb.options]       # etiket: "★ #87 · R12 · awp · 0.676"
    assert set(opts) <= set(fe["kill_id"].to_list())
    assert int(sb.value) in fe["kill_id"].to_list()                      # olay listesi aynı kümeden
    assert any(f"{fe.height} / {ev.height} olay" in x.value for x in at.markdown)
    assert [x.value for x in at.markdown if 'class="pcard' in x.value] == cards_before   # skor/öncelik aynı

    # filtre açıkken zaman çizelgesinden başka oyuncunun olayı (kill_id ile)
    other = fe.filter(pl.col("attacker") != player)
    if other.height:
        row = other.row(0, named=True)
        at.session_state[f"tl_pick_{demo}"] = V.timeline_pick([{"customdata": row["kill_id"]}], ev)
        at.session_state[f"tl_nonce_{demo}"] = 7
        at.run()
        assert at.session_state[f"player_{demo}"] == row["attacker"]
        assert at.selectbox(key=f"ev_{demo}_{row['attacker']}").value == row["kill_id"]


@pytest.mark.skipif(not V.list_analyses(), reason="analiz yok")
def test_app_filter_empty_state_and_dropped_selection():
    from streamlit.testing.v1 import AppTest
    E = V.list_analyses()
    d = V.load(E[V.default_index(E)])
    demo, ev = d["meta"]["demo_id"], d["events"]
    at = AppTest.from_file(APP, default_timeout=180).run()
    player = at.session_state[f"player_{demo}"]
    # seçili olay (oyuncunun en yüksek skorlu olayı) filtre dışında kalsın ama oyuncunun uyan olayı olsun
    mine = ev.filter(pl.col("attacker") == player).sort("event_score", descending=True)
    top = mine["kill_id"][0]
    at.selectbox(key=f"ev_{demo}_{player}").set_value(top).run()
    thr = V.aim_near_thresholds()
    flt = next((f for f in V.FILTERS
                if (ids := V.filter_events(mine, [f], 0, thr)["kill_id"].to_list()) and top not in ids), None)
    if flt is not None:
        ids = V.filter_events(mine, [flt], 0, thr)["kill_id"].to_list()
        at.session_state[f"flt_{demo}"] = [flt]
        at.run()
        assert not at.exception
        assert any("filtre dışında kaldı" in c.value for c in at.caption)
        assert at.selectbox(key=f"ev_{demo}_{player}").value == ids[0]     # filtreli en yüksek skorlu
    # hiçbir olayın uymadığı kombinasyon → boş durum mesajı, hata yok, temizle butonu
    at.session_state[f"flt_{demo}"] = list(V.FILTERS)
    at.session_state[f"fltp_{demo}"] = 99
    at.run()
    assert not at.exception
    assert V.filter_events(ev, list(V.FILTERS), 99, V.aim_near_thresholds()).height == 0
    assert any("Seçili filtrelere uyan olay yok" in i.value for i in at.info)
    assert _timeline_ids(at) is None or _timeline_ids(at) == []
    next(b for b in at.button if b.key == f"fltc2_{demo}").click().run()
    assert not at.exception and at.session_state[f"flt_{demo}"] == [] and at.session_state[f"fltp_{demo}"] == 0
    assert _timeline_ids(at) == sorted(ev["kill_id"].to_list())
