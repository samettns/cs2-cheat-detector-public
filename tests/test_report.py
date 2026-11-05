"""Faz 3.5 HTML inceleme raporu (src/report_export.py).

Yalnız mevcut çıktılar; zorunlu gizlilik denetimi; uyarı + köken notu en üstte; sayılar player_scores ile aynı; önemli
olaylar yalnız mevcut bayraklardan; en çok 6 radar görseli; radar yoksa boş zemin; CS2CD etiketi ayrı bölüm;
deterministik; arayüzde "Rapor oluştur" → indir.
"""
import json
import re
import sys
from html.parser import HTMLParser
from pathlib import Path

import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
import report_export as RE  # noqa: E402
import viewer_data as V  # noqa: E402

APP = str(Path(__file__).parent.parent / "app" / "app.py")
FIXED = "2026-01-01 00:00"
E_ALL = V.list_analyses()


def _html(e, **kw):
    return RE.build_report(Path(e["work"]), generated_at=FIXED, **kw)


def _by(pred):
    return next((e for e in E_ALL if pred(e)), None)


@pytest.mark.skipif(not E_ALL, reason="analiz yok")
@pytest.mark.parametrize("e", E_ALL, ids=[e["demo_id"] for e in E_ALL])
def test_report_sections_privacy_and_numbers(e):
    html = _html(e)
    assert RE.privacy_problems_html(html) == []
    HTMLParser().feed(html)                                                   # ayrıştırılabilir
    top = html.index("1. Analiz özeti")
    assert html.index(RE.DISCLAIMER) < top and html.index(RE.PROVENANCE) < top
    for sec in ["2. Model v1 bilgisi", "3. Demo düzeyi dağılım bilgisi", "4. İnceleme öncelikleri",
                "5. Oyuncular (model çıktısı)", "6. Davranış özetleri", "7. Seçilmiş önemli olaylar",
                "Model sınırlılıkları", "Round bilgileri"]:
        assert sec in html, sec
    d = V.load(e)
    for r in d["players"].iter_rows(named=True):
        assert f"<b>{r['player']}</b>" in html
        if r["player_raw_score"] is not None:
            assert f"{r['player_raw_score']:.3f}" in html
    for k in V.PRIORITY_ORDER:
        n = d["players"].filter(pl.col("review_priority") == k).height
        assert f"<b>{n}</b>{k}" in html
    assert html.count("data:image/jpeg;base64,") <= RE.MAX_EVENT_IMAGES
    assert not re.search(r"\bP(10|[1-9])\b", re.sub(r"data:image/[a-z]+;base64,[A-Za-z0-9+/=]+", "", html))   # yalnız Player_N


@pytest.mark.skipif(not E_ALL, reason="analiz yok")
def test_important_events_only_existing_flags_and_deterministic():
    for e in E_ALL:
        d = V.load(e)
        imp = RE.important_events(d)
        assert imp.height <= RE.MAX_EVENTS
        top_ids = {int(k) for r in d["players"].iter_rows(named=True)
                   if r["review_priority"] in ("yüksek inceleme önceliği", "orta inceleme önceliği")
                   for k in (r.get("top_event_ids") or [])}
        for r in imp.iter_rows(named=True):
            assert r["review_point"] or r["kill_id"] in top_ids
        rp = imp.filter(pl.col("review_point").fill_null(False))
        assert rp["event_score"].to_list() == sorted(rp["event_score"].to_list(), reverse=True)
        assert imp["kill_id"].to_list() == RE.important_events(d)["kill_id"].to_list()
    e = E_ALL[0]
    assert _html(e) == _html(e)                                                # aynı girdi + tarih → aynı HTML


def test_distribution_box_hltv_vs_valve():
    hltv = _by(lambda e: e.get("source") == "hltv")
    valve = _by(lambda e: e.get("source") == "valve")
    if not (hltv and valve):
        pytest.skip("HLTV/Valve analizi yok")
    assert 'class="dist"' in _html(hltv) and "Dağılım farkı" in _html(hltv)
    assert 'class="dist"' not in _html(valve) and "olağan aralığında" in _html(valve)


def test_cs2cd_label_section_separate():
    cs = _by(lambda e: e["mode"] == "cs2cd_oof")
    other = _by(lambda e: e["mode"] != "cs2cd_oof")
    if not cs:
        pytest.skip("CS2CD analizi yok")
    html = _html(cs)
    assert RE.LABEL_TITLE == "Veri seti etiketi — model çıktısı değildir"
    i = html.index(RE.LABEL_TITLE)
    assert html.index("Model sınırlılıkları") < i                              # model bölümlerinden sonra, ayrı
    assert 'class="labels"' in html and "CC-BY-4.0" in html
    sec5 = html[html.index("5. Oyuncular"):html.index("6. Davranış")]
    assert "Veri seti etiketi" not in sec5                                     # model tablosuna karışmaz
    if other:
        assert RE.LABEL_TITLE not in _html(other)


def test_comparison_optional():
    e = E_ALL[0] if E_ALL else pytest.skip("analiz yok")
    d = V.load(e)
    order = V.player_order(d["players"])
    html = _html(e, compare=(order[0], order[1]))
    assert "Oyuncu karşılaştırması" in html and V.COMPARE_NOTE in html
    assert "Oyuncu karşılaştırması" not in _html(e)
    with pytest.raises(ValueError):
        _html(e, compare=(order[0], "Player_999"))


def test_report_without_radar_assets(monkeypatch):
    e = _by(lambda e: e["mode"] == "faz2") or (E_ALL[0] if E_ALL else None)
    if not e:
        pytest.skip("analiz yok")
    monkeypatch.setattr(V.R, "load_info", lambda m, radar_dir=None: None)
    html = _html(e)
    assert "Harita görseli bulunamadı" in html and "data:image/jpeg;base64," in html


def test_privacy_checks_catch_forbidden_content(monkeypatch):
    fake_id = "7656119" + "8" * 10                    # parçalı yazıldı (pre-commit kancası gerçek ID arar)
    assert "Steam ID deseni" in RE.privacy_problems_html(f"<p>{fake_id}</p>")
    assert ".dem dosya adı" in RE.privacy_problems_html("<p>mac.dem</p>")
    assert "yerel dosya yolu" in RE.privacy_problems_html(r"<p>C:\Users\x</p>")
    assert "kimlik eşlemesi / steamid" in RE.privacy_problems_html("<p>a.ids.json</p>")
    assert RE.privacy_problems_html('<img src="data:image/jpeg;base64,QzpcVXNlcnM=">') == []
    e = E_ALL[0] if E_ALL else pytest.skip("analiz yok")
    real_load = V.load

    def bad_load(entry):
        d = real_load(entry)
        d["players"] = d["players"].with_columns(pl.lit("GercekIsim").alias("player"))
        return d
    monkeypatch.setattr(V, "load", bad_load)
    with pytest.raises(RE.ReportPrivacyError):
        _html(e)


@pytest.mark.skipif(not E_ALL, reason="analiz yok")
def test_app_report_button_and_download():
    from streamlit.testing.v1 import AppTest
    E = V.list_analyses()
    demo = E[V.default_index(E)]["demo_id"]
    at = AppTest.from_file(APP, default_timeout=180).run()
    next(b for b in at.button if b.key == f"rep_make_{demo}").click().run()
    assert not at.exception
    html = at.session_state[f"rep_html_{demo}"]
    assert RE.DISCLAIMER in html and RE.privacy_problems_html(html) == []
    assert any("KB · tek dosya" in c.value for c in at.caption)


# ---------------------------------------------------------------- görsel/okunabilirlik cilası
TOP1 = "Bu skor kalibre edilmiş hile olasılığı değildir."
TOP2 = ("Bu rapor mevcut Model v1 analiz çıktılarından oluşturulmuştur; rapor üretimi sırasında yeni model çıkarımı, "
        "yeni skor veya yeni eşik uygulanmaz.")


@pytest.mark.skipif(not E_ALL, reason="analiz yok")
@pytest.mark.parametrize("e", E_ALL, ids=[e["demo_id"] for e in E_ALL])
def test_limitations_split_keeps_every_item_and_turkish_terms(e):
    d = V.load(e)
    lims = V.limitations(e["mode"], d["result"])
    main, extra = RE.split_limitations(lims)
    assert sorted(main + extra) == sorted(lims) and 5 <= len(main) <= 6     # hiçbir madde silinmedi
    html = _html(e)
    assert "Ek sınırlılıklar" in html
    for x in lims:
        assert RE.tr_terms(x).replace("'", "&#39;").replace('"', "&#34;")[:60] in html
    tr = "Profesyonel / yüksek beceri yanlış alarm riski (professional/high-skill false-positive risk)"
    assert html.count("professional/high-skill false-positive risk") == html.count(tr) >= 1   # İngilizce tek başına yok
    assert html.index(TOP1) < html.index("1. Analiz özeti") and html.index(TOP2) < html.index("1. Analiz özeti")
    assert "Ham demo dosyası, Steam ID, gerçek oyuncu adı ve kimlik eşlemesi bu rapora girmez." in html
    assert html.count('class="keep"') >= 8                                   # başlık + ilk içerik birlikte
    assert 'class="compact"' in html                                          # davranış tablosu kompakt


def test_tr_terms_idempotent():
    t = "İlgili sınırlılık: professional/high-skill false-positive risk (notes)."
    once = RE.tr_terms(t)
    assert once == RE.tr_terms(once) and once.startswith("İlgili sınırlılık: Profesyonel / yüksek beceri")
    assert RE.tr_terms("Pre-FOV açısı (rakip görünmeden ~190 ms önce)") == \
        "Görüş öncesi nişan açısı (pre-FOV; rakip görünmeden ~190 ms önce)"


def test_image_footnote_small_for_events_without_image():
    e = next((x for x in E_ALL if RE.important_events(V.load(x)).height > RE.MAX_EVENT_IMAGES), None)
    if not e:
        pytest.skip("6'dan fazla önemli olayı olan analiz yok")
    html = _html(e)
    assert 'class="imgnote"' in html and "Radar görseli yalnız ilk 6 olay için eklenir." in html
    assert html.count("data:image/jpeg;base64,") == RE.MAX_EVENT_IMAGES
