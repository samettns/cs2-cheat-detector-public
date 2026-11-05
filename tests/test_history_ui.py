"""Faz 3.5 analiz geçmişi, yükleme ilerlemesi ve sunum modu — yalnız gösterim.

- history_rows: sayılar player_scores ile aynı; örnek paket tarihsiz ("örnek paket"); sıralama; roller
- progress_view: 7 adım mevcut işaretlere eşlenir; bu aşamada geçen süre dosya zamanlarından; yüzde/ETA yok
- geçmiş ölçüm aralığı yalnız yerel analizlerden ve "kalan süre tahmini değildir" etiketli
- arayüz: geçmiş ekranı + "Aç"; ilerleme paneli; sunum modu skor kartlarını değiştirmez
"""
import json
import os
import re
import sys
import time
from pathlib import Path

import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
import upload_flow as U  # noqa: E402
import viewer_data as V  # noqa: E402

APP = str(Path(__file__).parent.parent / "app" / "app.py")
E_ALL = V.list_analyses()


# ---------------------------------------------------------------- geçmiş
@pytest.mark.skipif(not E_ALL, reason="analiz yok")
def test_history_rows_counts_dates_and_order():
    rows = V.history_rows(E_ALL)
    assert {r["demo_id"] for r in rows} == {e["demo_id"] for e in E_ALL}
    for r, e in ((r, next(e for e in E_ALL if e["demo_id"] == r["demo_id"])) for r in rows):
        P = V.load(e)["players"]
        cnt = {k: P.filter(pl.col("review_priority") == k).height for k in V.PRIORITY_ORDER}
        assert (r["high"], r["mid"], r["low"], r["insufficient"]) == tuple(cnt[k] for k in V.PRIORITY_ORDER)
        assert r["rounds"] == V.load(e)["meta"].get("rounds")
        if V._is_sample(Path(e["work"])):
            assert r["date"] == V.SAMPLE_DATE and r["timestamp"] is None and "örnek paket" in r["role"]
        else:
            assert re.fullmatch(r"\d{4}-\d\d-\d\d \d\d:\d\d", r["date"])
    ts = [r["timestamp"] for r in rows]
    dated = [t for t in ts if t is not None]
    assert dated == sorted(dated, reverse=True)
    assert all(t is None for t in ts[len(dated):])                         # örnek paket en sonda


def test_history_sample_pack_never_shows_date():
    S = V.list_analyses(V.SAMPLES_ROOT)
    if not S:
        pytest.skip("örnek paket yok")
    rows = V.history_rows(S)
    assert all(r["date"] == "örnek paket" for r in rows)
    roles = " ".join(r["role"] for r in rows)
    assert "örnek:" in roles


# ---------------------------------------------------------------- ilerleme
def _touch(path: Path, t: float):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{}", encoding="utf-8")
    os.utime(path, (t, t))


def test_progress_view_steps_and_measured_stage_time(tmp_path):
    a, c, did, t0 = tmp_path / "analysis", tmp_path / "cache", "x1", 1_000_000.0
    job = dict(state="running", started=t0)
    pv = U.progress_view(did, job, now=t0 + 5, analysis_root=a, cache_root=c)
    assert [s["label"] for s in pv["steps"]] == [lab for _, lab in U.DISPLAY_STEPS] and pv["n"] == 7
    assert pv["current"] == "Demo okunuyor" and pv["index"] == 1 and pv["elapsed_stage"] == 5
    _touch(c / did / "adapter_meta.json", t0 + 10)
    _touch(a / did / "_stages" / "aim.parquet", t0 + 20)
    pv = U.progress_view(did, job, now=t0 + 200, analysis_root=a, cache_root=c)
    assert pv["current"] == "Görünürlük hesaplanıyor" and pv["index"] == 3
    assert pv["elapsed_stage"] == pytest.approx(180) and pv["elapsed_total"] == pytest.approx(200)
    assert [s["state"] for s in pv["steps"]] == ["done", "done", "done", "running", "pending", "pending", "pending"]
    pv = U.progress_view(did, dict(state="failed", started=t0), now=t0 + 300, analysis_root=a, cache_root=c)
    assert pv["steps"][3]["state"] == "failed"
    pv = U.progress_view(did, dict(state="done", started=t0), now=t0 + 400, analysis_root=a, cache_root=c)
    assert all(s["state"] == "done" for s in pv["steps"]) and pv["current"] is None
    assert not any(k in pv for k in ("percent", "eta", "remaining"))         # yüzde / kalan süre üretilmez


def test_past_stage_text_only_local_measurements(tmp_path):
    assert U.past_stage_text("gorunurluk", tmp_path) is None                  # ölçüm yoksa hiçbir şey
    for i, sec in enumerate([120.0, 480.0]):
        d = tmp_path / f"d{i}"
        d.mkdir()
        (d / "metadata.json").write_text(json.dumps({"run": {"seconds_by_stage": {"gorunurluk": sec}}}),
                                         encoding="utf-8")
    txt = U.past_stage_text("gorunurluk", tmp_path)
    assert txt == ("Bu makinedeki önceki analizlerde görünürlük adımı 2.0–8.0 dk sürdü (2 analiz). "
                   "Bu bir kalan süre tahmini değildir.")
    assert "%" not in txt


# ---------------------------------------------------------------- arayüz
def _cards(at):
    return [x.value for x in at.markdown if 'class="pcard' in x.value]


@pytest.mark.skipif(not E_ALL, reason="analiz yok")
def test_app_history_view_and_open():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(APP, default_timeout=180).run()
    next(b for b in at.sidebar.button if b.key == "hist_open").click().run()
    assert not at.exception and at.session_state["show_history"] is True
    md = " ".join(x.value for x in at.markdown)
    assert "Analiz geçmişi" in md and not _cards(at)                       # inceleme ekranı çizilmez
    target = next(r for r in V.history_rows(E_ALL) if r["timestamp"] is None) if any(
        V._is_sample(Path(e["work"])) for e in E_ALL) else V.history_rows(E_ALL)[-1]
    assert {b.key for b in at.button} >= {f"hist_{e['demo_id']}" for e in E_ALL}
    next(b for b in at.button if b.key == f"hist_{target['demo_id']}").click().run()
    assert not at.exception and at.session_state["show_history"] is False
    assert at.query_params["demo"] == [target["demo_id"]] or at.query_params["demo"] == target["demo_id"]
    assert _cards(at)                                                      # inceleme ekranı geri geldi


@pytest.mark.skipif(not E_ALL, reason="analiz yok")
def test_app_job_progress_panel_no_percent(monkeypatch):
    from streamlit.testing.v1 import AppTest
    t0 = time.time() - 125
    fake = dict(steps=[dict(key=k, label=lab, state=s) for (k, lab), s in zip(
        U.DISPLAY_STEPS, ["done", "done", "done", "running", "pending", "pending", "pending"])],
        index=3, n=7, current="Görünürlük hesaplanıyor", current_key="gorunurluk", elapsed_total=125.0,
        elapsed_stage=95.0)
    monkeypatch.setattr(U, "running_job", lambda *a, **k: "fakejob")
    monkeypatch.setattr(U, "read_job", lambda *a, **k: dict(did="fakejob", state="running", started=t0))
    monkeypatch.setattr(U, "progress_view", lambda *a, **k: fake)
    monkeypatch.setattr(U, "past_stage_text", lambda *a, **k: "Bu makinedeki önceki analizlerde görünürlük adımı "
                                                               "1.7–9.8 dk sürdü (9 analiz). Bu bir kalan süre "
                                                               "tahmini değildir.")
    at = AppTest.from_file(APP, default_timeout=180).run()
    assert not at.exception
    md = " ".join(x.value for x in at.markdown)
    assert "Aşama 4 / 7" in md and "Görünürlük hesaplanıyor" in md and "bu aşamada 1 dk 35 sn" in md
    assert "Bu bir kalan süre tahmini değildir." in md
    assert not at.get("progress")                                         # sürekli yüzde çubuğu yok
    head = next(x.value for x in at.markdown if 'class="jobtime"' in x.value)
    assert "%" not in head and "kalan" not in head


@pytest.mark.skipif(not E_ALL, reason="analiz yok")
def test_app_present_mode_display_only():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(APP, default_timeout=180).run()
    cards = _cards(at)
    player = at.session_state[f"player_{V.list_analyses()[V.default_index(V.list_analyses())]['demo_id']}"]
    at.session_state["present_mode"] = True
    at.run()
    assert not at.exception
    assert any("html {font-size: 17px" in x.value for x in at.markdown)
    assert _cards(at) == cards                                            # skor/öncelik kartları aynı
    radar = json.loads(at.get("plotly_chart")[0].proto.spec)
    assert radar["layout"]["height"] == 860                               # radar daha baskın
    demo = V.list_analyses()[V.default_index(V.list_analyses())]["demo_id"]
    assert at.session_state[f"player_{demo}"] == player
    at.session_state["present_mode"] = False
    at.run()
    assert json.loads(at.get("plotly_chart")[0].proto.spec)["layout"]["height"] == 730


@pytest.mark.skipif(not E_ALL, reason="analiz yok")
def test_filter_expander_label_constant_when_active():
    """Final regresyonda bulunan hata: başlık "· 1 aktif" ile değişince Streamlit bölümü kapatıyordu."""
    from streamlit.testing.v1 import AppTest
    demo = V.list_analyses()[V.default_index(V.list_analyses())]["demo_id"]
    at = AppTest.from_file(APP, default_timeout=180).run()
    labels0 = [x.label for x in at.expander]
    at.session_state[f"flt_{demo}"] = ["review_point"]
    at.run()
    assert not at.exception and "Olay filtreleri" in [x.label for x in at.expander]
    assert [x.label for x in at.expander] == labels0
