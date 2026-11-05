"""Faz 3.5 oyuncu karşılaştırma (viewer_data.compare_players, app "Oyuncu karşılaştırma" görünümü).

YALNIZ GÖSTERİM: değerler mevcut çıktılar / modele giren bütün öldürmelerin ortalaması; yüzdelikler "Neden bu öncelik?"
kartlarıyla tutarlı; yön yalnız "A/B daha yüksek"; karar dili yok; olay filtreleri etkilemez; görünüm değişimi inceleme
durumunu (oyuncu, olay, filtre, radar modu, skor kartları) değiştirmez; "İncelemede aç" doğru oyuncuyu açar.
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
import viewer_data as V  # noqa: E402

APP = str(Path(__file__).parent.parent / "app" / "app.py")
BANNED = ("hileli", "hileci", "şüphe", "daha iyi", "daha kötü", "suçlu")


def _entry():
    E = V.list_analyses()
    return E[V.default_index(E)] if E else None


def _two(d):
    order = V.player_order(d["players"])
    scored = [p for p in order if d["players"].filter(pl.col("player") == p)["player_raw_score"][0] is not None]
    return scored[0], scored[1]


@pytest.mark.skipif(_entry() is None, reason="analiz yok")
def test_values_are_means_over_all_model_kills():
    d = V.load(_entry())
    a, b = _two(d)
    rows = {r["key"]: r for r in V.compare_players(d, a, b, extra=True)}
    ev_a = d["events"].filter(pl.col("attacker") == a)
    for f in ["reaction_ms", "speed_max", "jitter", "still_ratio", "prefov_angle", "inv_angle_med", "inv_frac_lt10",
              "speed_mean", "speed_last_mean", "pitch_range_last", "shots", "after_move"]:
        s = ev_a[f].cast(pl.Float64).drop_nulls().drop_nans()
        assert rows[f]["a_n"] == s.len()
        assert rows[f]["a"] == pytest.approx(float(s.mean())) if s.len() else rows[f]["a"] is None
    pa = d["players"].filter(pl.col("player") == a).row(0, named=True)
    assert rows["player_raw_score"]["a"] == pa["player_raw_score"]
    assert rows["style_percentile"]["a"] == pa["style_percentile"]
    assert rows["analyzed_kills"]["a"] == pa["analyzed_kills"] == ev_a.height       # bütün modele giren öldürmeler
    wb = rows["wallbang"]
    assert wb["a_count"] == ev_a.filter(pl.col("wallbang") > 0).height
    assert wb["a_text"].startswith(f"{wb['a_count']} / {wb['a_n']} (%")             # sayı + oran
    sm = rows["thrusmoke"]
    assert sm["a_count"] == ev_a.filter(pl.col("thrusmoke")).height and "%" in sm["a_text"]


@pytest.mark.skipif(_entry() is None, reason="analiz yok")
def test_percentiles_consistent_with_explanation_cards():
    d = V.load(_entry())
    checked = 0
    for r in d["players"].iter_rows(named=True):
        tops = json.loads(r["player_top_features"] or "[]")
        if not tops:
            continue
        rows = {x["key"]: x for x in V.compare_players(d, r["player"], r["player"])}
        for t in tops:
            if t["feature"] not in rows:
                continue
            below = rows[t["feature"]]["a_pct"]
            # kart: +1 yön → temizlerin yüzde kaçından büyük (= below); −1 → yüzde kaçından küçük (≈ 100 − below)
            expect = below if t["direction"] > 0 else 100 - below
            assert abs(t["notable_pct"] - expect) <= 1.5, (r["player"], t["feature"])
            checked += 1
    assert checked >= 3


@pytest.mark.skipif(_entry() is None, reason="analiz yok")
def test_same_player_equal_and_neutral_language():
    d = V.load(_entry())
    a, _ = _two(d)
    rows = V.compare_players(d, a, a, extra=True)
    assert all(r["direction"] in ("eşit", "aynı", "karşılaştırılamaz") for r in rows)
    texts = " ".join([V.COMPARE_NOTE] + [str(r["label"]) + " " + str(r.get("note") or "") for r in rows]).lower()
    assert not any(w in texts for w in BANNED)
    notes = {r["key"]: r.get("note") for r in rows}
    assert notes["reaction_ms"] == "Düşük = daha kısa tepki"
    assert notes["prefov_angle"] == notes["inv_angle_med"] == "Düşük = hedefe daha yakın nişan"
    assert "wallhack anlamına gelmez" in notes["not_visible"]
    assert {r["direction"] for r in V.compare_players(d, *_two(d))} <= {
        "A daha yüksek", "B daha yüksek", "eşit", "aynı", "farklı", "karşılaştırılamaz"}
    # yeni skor / eşik alanı üretilmez
    assert all(not re.search(r"score|threshold|eşik", k) or k == "player_raw_score"
               for r in rows for k in [r["key"]])


def test_direction_rule():
    assert V._direction("1.0", "2.0", 1.0, 2.0) == "B daha yüksek"
    assert V._direction("2.0", "1.0", 2.0, 1.0) == "A daha yüksek"
    assert V._direction("1.0", "1.0", 1.001, 1.0) == "eşit"          # gösterilen değer aynıysa eşit
    assert V._direction("x", "bilinmiyor", 1.0, None) == "karşılaştırılamaz"


# ---------------------------------------------------------------- arayüz
def _cards(at):
    return [x.value for x in at.markdown if 'class="pcard' in x.value]


def _cmp_html(at):
    return "".join(x.value for x in at.markdown if 'class="cmp"' in x.value)


@pytest.mark.skipif(not V.list_analyses(), reason="analiz yok")
def test_compare_view_runs_for_every_analysis():
    from streamlit.testing.v1 import AppTest
    for e in V.list_analyses():
        at = AppTest.from_file(APP, default_timeout=180)
        at.query_params["demo"] = e["demo_id"]
        at.run()
        at.session_state[f"view_{e['demo_id']}"] = "Oyuncu karşılaştırma"
        at.run()
        assert not at.exception, e["demo_id"]
        html = _cmp_html(at)
        assert "Model skoru (ortalama)" in html and "Wallbang öldürme" in html, e["demo_id"]
        assert not any(w in html.lower() for w in BANNED)


@pytest.mark.skipif(_entry() is None, reason="analiz yok")
def test_view_switch_preserves_inspection_state_and_ignores_filters():
    from streamlit.testing.v1 import AppTest
    d = V.load(_entry())
    demo = d["meta"]["demo_id"]
    at = AppTest.from_file(APP, default_timeout=180).run()
    player = at.session_state[f"player_{demo}"]
    ev_key = f"ev_{demo}_{player}"
    second = int(re.search(r"#(\d+)", at.selectbox(key=ev_key).options[1]).group(1))
    at.selectbox(key=ev_key).set_value(second).run()
    at.session_state[f"mode_{demo}"] = "Round Oynat"
    at.run()
    at.session_state[f"flt_{demo}"] = ["review_point"]
    at.run()
    cards = _cards(at)
    # karşılaştırma: filtre açıkken de özetler bütün öldürmelerden
    at.session_state[f"view_{demo}"] = "Oyuncu karşılaştırma"
    at.run()
    assert not at.exception
    a = at.selectbox(key=f"cmp_a_{demo}").value
    b = at.selectbox(key=f"cmp_b_{demo}").value
    assert a == player
    expected = V.compare_players(d, a, b)
    html = _cmp_html(at)
    for r in expected:
        if r["kind"] != "model":
            assert r["a_text"] in html and r["b_text"] in html
    # dönüş: oyuncu, olay, filtre, radar modu, skor kartları aynı
    at.session_state[f"view_{demo}"] = "İnceleme"
    at.run()
    assert not at.exception
    assert at.session_state[f"player_{demo}"] == player
    assert at.selectbox(key=ev_key).value == second
    assert at.session_state[f"flt_{demo}"] == ["review_point"]
    assert at.session_state[f"mode_{demo}"] == "Round Oynat"
    assert _cards(at) == cards


@pytest.mark.skipif(_entry() is None, reason="analiz yok")
def test_open_in_inspect_selects_player():
    from streamlit.testing.v1 import AppTest
    d = V.load(_entry())
    demo = d["meta"]["demo_id"]
    at = AppTest.from_file(APP, default_timeout=180).run()
    at.session_state[f"view_{demo}"] = "Oyuncu karşılaştırma"
    at.run()
    b = at.selectbox(key=f"cmp_b_{demo}").value
    next(x for x in at.button if x.key == f"cmp_open_b_{demo}").click().run()
    assert not at.exception
    assert at.session_state[f"view_{demo}"] == "İnceleme" and at.session_state[f"player_{demo}"] == b
    assert any('class="pcard sel' in x.value and f">{b}<" in x.value for x in at.markdown)
