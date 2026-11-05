"""Faz 3.5 round oynatma (src/playback.py, app Round Oynat modu) ve radar etiket sadeleştirme testleri.

- Hz: varsayılan 8, CS2_PLAYBACK_HZ=4 güvenli yedek, başka değer → 8; kare sayısı Hz'e göre
- iz ≤ 1 sn; etiketler yalnız Player_N (+n); görünen iki etiket 30 px'ten yakın değil; hover'da tam ad
- kaydırıcıda ≤ 12 süre etiketi
- arayüz: mod geçişi skor/olay/oyuncu/filtre durumunu değiştirmez; oynatmadayken zaman çizelgesi tıklaması Olay İncele'ye
  döndürür; statik radarda P1 biçimi yok
"""
import json
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
import playback as PB  # noqa: E402
import viewer_data as V  # noqa: E402

APP = str(Path(__file__).parent.parent / "app" / "app.py")
LABEL = re.compile(r"^(Player_\d+( \+\d+)?)?$")
PB_LABEL = re.compile(r"^(Player_\d+( \+\d+)?| )$")      # oynatma: gizli etiket = " " (redraw=False için)


def _entry():
    E = V.list_analyses()
    return E[V.default_index(E)] if E else None


# ---------------------------------------------------------------- etiket sadeleştirme
def test_declutter_keeps_points_merges_close_labels():
    xs, ys = [0, 10, 100, 105, 500], [0, 0, 0, 0, 0]
    names = ["Player_2", "Player_10", "Player_3", "Player_4", "Player_5"]
    texts, res = V.declutter_labels(xs, ys, names)
    assert texts == ["Player_2 +1", "", "Player_3 +1", "", "Player_5"]       # doğal sıra: Player_2 önce
    assert len(texts) == len(names) and res == []                          # her nokta için bir giriş (nokta kalır)
    texts, res = V.declutter_labels([0, 200], [0, 0], ["Player_1", "Player_2"], reserved=[(5, 5)])
    assert texts == ["", "Player_2"] and res == [1]                        # seçili oyuncu etiketi "+1" alır
    shown = [(x, t) for x, t in zip(xs, V.declutter_labels(xs, ys, names)[0]) if t]
    assert all(abs(a[0] - b[0]) >= V.LABEL_MIN_PX for i, a in enumerate(shown) for b in shown[i + 1:])


# ---------------------------------------------------------------- figür
def test_playback_hz_env(monkeypatch):
    monkeypatch.delenv("CS2_PLAYBACK_HZ", raising=False)
    assert PB.playback_hz() == 8
    monkeypatch.setenv("CS2_PLAYBACK_HZ", "4")
    assert PB.playback_hz() == 4
    for bad in ("2", "16", "abc", ""):
        monkeypatch.setenv("CS2_PLAYBACK_HZ", bad)
        assert PB.playback_hz() == 8
    # çok uzun round → güvenli 4 Hz (eşik > 1000 kare); normal round 8 Hz
    monkeypatch.delenv("CS2_PLAYBACK_HZ", raising=False)
    assert PB.LONG_ROUND_FRAMES == 1000
    assert PB.playback_hz(920) == 8 and PB.playback_hz(1000) == 8 and PB.playback_hz(1001) == 4
    assert PB.playback_hz(1174) == 4


@pytest.mark.skipif(_entry() is None, reason="analiz yok")
def test_playback_figure_frames_trail_labels_slider():
    work = Path(_entry()["work"])
    rt = PB.round_table(work).sort("frames_8hz")
    rnd = rt["round"][-1]                                                  # en uzun round
    f8, t8 = PB.build_playback_figure(work, rnd, 8)
    f4, t4 = PB.build_playback_figure(work, rnd, 4)
    assert t8["frames"] == rt["frames_8hz"][-1] and t4["frames"] == -(-t8["frames"] // 2)
    assert t8["trail_s"] == PB.PLAYBACK_TRAIL_S == 1.0 and t8["hz"] == 8
    assert len(f8["frames"]) == t8["frames"] and f8["layout"]["updatemenus"][0]["buttons"][1:] and \
        [b["label"] for b in f8["layout"]["updatemenus"][0]["buttons"]] == ["⏸", "0.5×", "1×", "2×", "4×"]
    labels = [s["label"] for s in f8["layout"]["sliders"][0]["steps"] if s["label"]]
    assert 2 <= len(labels) <= PB.SLIDER_LABELS and labels[0] == "0:00"
    max_trail_pts = int(PB.PLAYBACK_TRAIL_S * 8) + 1
    for fr in f8["frames"][::25]:
        d = fr["data"]
        for tr in d[:2]:                                                   # iz: oyuncu başına ≤ 1 sn
            seg, n = [], 0
            for x in tr["x"] + [None]:
                if x is None:
                    seg.append(n)
                    n = 0
                else:
                    n += 1
            assert max(seg) <= max_trail_pts
        pts = []
        for tr in d[2:4]:                                                  # oyuncular: etiket + hover
            assert all(PB_LABEL.match(t) for t in tr["text"])
            assert all(re.fullmatch(r"Player_\d+", h) for h in tr["hovertext"])
            pts += [(x, y, t) for x, y, t in zip(tr["x"], tr["y"], tr["text"]) if t.strip()]
        for i, a in enumerate(pts):
            for b in pts[i + 1:]:
                assert (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 >= V.LABEL_MIN_PX ** 2
        assert re.fullmatch(r"\d+:\d\d · tick \d+", d[6]["text"][0])      # saat göstergesi (iz 6; 7–10 diğer kat)
    json.dumps(f8)                                                         # serileştirilebilir


# ---------------------------------------------------------------- arayüz
def _cards(at):
    return [x.value for x in at.markdown if 'class="pcard' in x.value]


def _has_frames(at) -> bool:
    return any(json.loads(c.proto.spec).get("frames") for c in at.get("plotly_chart"))


@pytest.mark.skipif(_entry() is None, reason="analiz yok")
def test_mode_switch_preserves_selection_and_scores():
    from streamlit.testing.v1 import AppTest
    d = V.load(_entry())
    demo = d["meta"]["demo_id"]
    at = AppTest.from_file(APP, default_timeout=180).run()
    assert not at.exception and at.session_state[f"mode_{demo}"] == "Olay İncele" and not _has_frames(at)
    player = at.session_state[f"player_{demo}"]
    ev_key = f"ev_{demo}_{player}"
    other = [o for o in at.selectbox(key=ev_key).options][1]
    at.selectbox(key=ev_key).set_value(int(re.search(r"#(\d+)", other).group(1))).run()
    chosen, cards = at.selectbox(key=ev_key).value, _cards(at)
    at.session_state[f"flt_{demo}"] = []
    at.session_state[f"mode_{demo}"] = "Round Oynat"
    at.run()
    assert not at.exception and _has_frames(at)                           # oynatma figürü
    assert any("kare" in x.value and "Hz" in x.value for x in at.markdown)  # teknik satır
    assert at.session_state[f"player_{demo}"] == player and at.selectbox(key=ev_key).value == chosen
    assert _cards(at) == cards                                            # skor/öncelik kartları aynı
    at.session_state[f"mode_{demo}"] = "Olay İncele"
    at.run()
    assert not at.exception and not _has_frames(at)
    assert at.session_state[f"player_{demo}"] == player and at.selectbox(key=ev_key).value == chosen
    assert _cards(at) == cards


@pytest.mark.skipif(_entry() is None, reason="analiz yok")
def test_timeline_click_in_playback_returns_to_event_mode():
    from streamlit.testing.v1 import AppTest
    d = V.load(_entry())
    demo, ev = d["meta"]["demo_id"], d["events"]
    at = AppTest.from_file(APP, default_timeout=180).run()
    at.session_state[f"mode_{demo}"] = "Round Oynat"
    at.run()
    row = ev.sort("event_score").row(0, named=True)
    at.session_state[f"tl_pick_{demo}"] = V.timeline_pick([{"customdata": row["kill_id"]}], ev)
    at.session_state[f"tl_nonce_{demo}"] = 3
    at.run()
    assert not at.exception and at.session_state[f"mode_{demo}"] == "Olay İncele"
    assert at.session_state[f"player_{demo}"] == row["attacker"]
    assert at.selectbox(key=f"ev_{demo}_{row['attacker']}").value == row["kill_id"]


@pytest.mark.skipif(_entry() is None, reason="analiz yok")
def test_static_radar_uses_player_n_labels_only():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(APP, default_timeout=180).run()
    radar = json.loads(at.get("plotly_chart")[0].proto.spec)
    texts = [t for tr in radar["data"] if tr.get("mode", "").endswith("text") for t in (tr.get("text") or [])]
    assert texts and all(LABEL.match(t) for t in texts)
    assert not any(re.fullmatch(r"P\d+", t) for t in texts)
