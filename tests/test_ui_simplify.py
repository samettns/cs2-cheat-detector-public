"""Final UI sadeleştirme testleri — yalnız gösterim düzeni.

- sağ panel ilk bakış: Player_N + rozet + model skoru, stil içi yüzdelik, inceleme noktası, modele giren öldürme
- "Neden bu öncelik?": en fazla 3 kısa neden; değerler metric_cards ile aynı (yeni hesap yok)
- taşınan metinler silinmedi: "Detaylı metrikler" / "Olay ayrıntıları" içinde
- oyuncu kartı: tek özet satırı; stil üzerine gelince
"""
import json
import re
import sys
from pathlib import Path

import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
import viewer_data as V  # noqa: E402

APP = str(Path(__file__).parent.parent / "app" / "app.py")
FIRST_LOOK = ["Model skoru (ortalama)", "Stil içi yüzdelik", "İnceleme noktası", "Modele giren öldürme"]


def _open(demo_id):
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(APP, default_timeout=180)
    at.query_params["demo"] = demo_id
    return at.run()


def _kv_keys(html: str) -> list[str]:
    return re.findall(r'<div class="k">([^<]*)</div>', html)


def _expander(at, label):
    return next(x for x in at.expander if x.label == label)


def test_reason_lines_top3_same_values():
    tf = json.dumps([dict(feature=f, value=v, notable_pct=p, text="")
                     for f, v, p in (("reaction_ms", 117.0, 97.0), ("prefov_angle", 4.0, 91.0),
                                     ("jitter", 0.5, 60.0), ("shots", 3.0, 80.0))])
    cards = V.metric_cards(tf, n=4)
    r = V.reason_lines(cards)
    assert len(r) == 3 and [x["name"] for x in r] == [c["name"] for c in cards[:3]]
    assert [x["value"] for x in r] == [c["value"] for c in cards[:3]] and r[2]["typical"]
    assert V.reason_lines([]) == []


@pytest.mark.skipif(not V.list_analyses(), reason="analiz yok")
def test_right_panel_first_look_and_moved_details():
    e = V.list_analyses()[V.default_index(V.list_analyses())]
    at = _open(e["demo_id"])
    assert not at.exception
    inside = {x.value for lab in ("Detaylı metrikler", "Olay ayrıntıları") for x in _expander(at, lab).markdown}
    top_kv = [x.value for x in at.markdown if 'class="kv"' in x.value and x.value not in inside]
    assert _kv_keys(top_kv[0]) == FIRST_LOOK                              # seçili oyuncu: yalnız 4 temel alan
    assert _kv_keys(top_kv[1]) == ["Round · tick", "Silah → kurban", "Olay skoru", "İnceleme noktası"]
    rs = [x.value for x in at.markdown if "Neden bu öncelik?" in x.value]
    assert rs and 1 <= rs[0].count('class="rsn"') <= 3 and 'class="mcard"' not in rs[0]
    det = " ".join(x.value for x in _expander(at, "Detaylı metrikler").markdown)
    assert "Oyuncu eşiği (yüksek)" in det and "Stil" in det and 'class="mcard"' in det
    assert any("temiz oyuncuların oyuncu ortalamaları" in c.value for c in _expander(at, "Detaylı metrikler").caption)
    od = " ".join(x.value for x in _expander(at, "Olay ayrıntıları").markdown)
    for k in ("Stil içi olay yüzdeliği", "Olay eşiği (stil içi %95)", "Wallbang", "Duman içinden",
              "Görünürlük (ray-cast)", "Tepki süresi", "Pre-FOV açısı", "İnceleme noktası nedir?"):
        assert k in od, k
    assert V.POINT_DEF in [c.value for c in _expander(at, "Olay ayrıntıları").caption]
    # oyuncu kartı: ad + rozet + tek özet satırı; stil title'da
    d = V.load(e)
    cards = [x.value for x in at.markdown if 'class="pcard' in x.value]
    for r in d["players"].iter_rows(named=True):
        c = next(c for c in cards if f">{r['player']}<" in c)
        assert c.count('class="sub"') == 1 and f'title="Stil: {r["style"] or V.UNKNOWN}"' in c


def _low_conf_entry():
    for e in V.list_analyses():
        P = V.load(e)["players"]
        if "low_confidence_style" in P.columns and P.filter(pl.col("low_confidence_style").fill_null(False)).height:
            return e, P.filter(pl.col("low_confidence_style").fill_null(False))["player"][0]
    return None, None


@pytest.mark.skipif(_low_conf_entry()[0] is None, reason="düşük güvenli oyuncu yok")
def test_low_confidence_text_kept_in_details():
    e, p = _low_conf_entry()
    at = _open(e["demo_id"])
    at.button(key=f"pick_{p}").click().run() if any(b.key == f"pick_{p}" for b in at.button) else None
    assert not at.exception
    assert any("⚠ düşük güven" in x.value for x in at.markdown)                       # önde kısa işaret
    assert any(V.LOW_CONF in c.value for c in _expander(at, "Detaylı metrikler").caption)   # tam metin silinmedi
