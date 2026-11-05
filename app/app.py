"""Faz 3: 2D analiz paneli (Streamlit + Plotly). Yalnız görüntüleme ve inceleme; model/skor mantığı DEĞİŞMEZ.

    .venv\\Scripts\\python.exe -m streamlit run app\\app.py

Düzen: sol = oyuncu kartları (inceleme önceliği, sonra stil içi yüzdelik), orta = 2D radar (ana unsur), sağ = seçili
oyuncu + seçili olay, alt = olay zaman çizelgesi. Büyük tablolar "Detaylı veriler"de, uzun uyarılar "Model
sınırlılıkları"nda. Önceden analiz edilmiş demolar (reports/analysis/) açılır; yeni .dem kenar çubuğundan
yüklenir (src/upload_flow.py: analiz ayrı süreçte; başarıdan sonra ham demo + kimlik eşlemesi silinir).
Renk dili: kırmızı = yüksek / seçili saldırgan, turuncu = orta, yeşil = düşük, sarı = seçili olay / tick,
mavi = kurban / referans, gri = diğer oyuncular. Terimler: model skoru, stil içi yüzdelik, inceleme önceliği,
inceleme noktası. "Hileci" hükmü verilmez.
"""
import html
import os
import sys
import time
from pathlib import Path

import plotly.graph_objects as go
import polars as pl
import streamlit as st
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)
import playback as PB      # noqa: E402
import radar as R          # noqa: E402
import score_demo as S     # noqa: E402
import upload_flow as U    # noqa: E402
import viewer_data as V    # noqa: E402

# ---------------------------------------------------------------- renk dili (tek yerde)
RED, ORANGE, GREEN, YELLOW, BLUE, GRAY = "#ff4d4f", "#f5a524", "#3fb68b", "#ffd400", "#3b8bff", "#8b949e"
ATTACKER, VICTIM, MARK = RED, BLUE, YELLOW
OTHER_T, OTHER_CT = "rgba(225,225,225,0.85)", "rgba(150,155,165,0.85)"    # takımlar: nötr gri tonları + simge
SMOKE, FIRE = "rgba(200,200,210,0.40)", "rgba(235,104,52,0.30)"
PRIO_COLOR = {"yüksek inceleme önceliği": RED, "orta inceleme önceliği": ORANGE,
              "düşük inceleme önceliği": GREEN, "yetersiz veri": GRAY}
LOW_CONF, SNIPER_LIMIT, RADAR_MISSING, PRO_FP_LIMIT, POINT_DEF = (   # ortak metinler (rapor da kullanır)
    V.LOW_CONF, V.SNIPER_LIMIT, V.RADAR_MISSING, V.PRO_FP_LIMIT, V.POINT_DEF)
MODE_EVENT, MODE_PLAY = "Olay İncele", "Round Oynat"      # radar alanının iki görünümü (yalnız görünüm)
VIEW_INSPECT, VIEW_COMPARE = "İnceleme", "Oyuncu karşılaştırma"   # sayfa görünümü
CMP_A, CMP_B = "#b98cff", "#2ec4b6"    # karşılaştırma: oyuncu renkleri (iyi/kötü anlamı yok)
JURY_TEXT = {   # "bu örnek neyi gösteriyor" (jury_set.json rolleri)
    "dogrulanmis_pozitif": ("Doğrulanmış pozitif örnek",
                            "CS2CD'nin hileli maçı; veri setinde hileci olarak doğrulanmış oyuncular var. Model bu "
                            "oyunculara hangi inceleme önceliğini veriyor? (Model etiketi görmez; skorlar OOF.)"),
    "hilecisiz": ("Hilecisiz maç örneği",
                  "CS2CD'nin hilecisiz veri setinden bir maç. Temiz oyuncuların inceleme önceliğinin nasıl "
                  "dağıldığını gösterir (yanlış alarm oranına bir örnek)."),
    "yanlis_alarm": ("Bilinen yanlış alarm örneği",
                     "Hilecisiz maçta sniper ağırlıklı temiz bir oyuncu yüksek inceleme önceliği alıyor. "
                     "Önceden tanımlı kuralla seçildi; sistemin sınırını göstermek için."),
}

st.set_page_config(page_title="CS2 Davranış Analizi", layout="wide", initial_sidebar_state="expanded")
st.markdown(f"""<style>
#MainMenu, footer, [data-testid="stStatusWidget"] {{visibility: hidden;}}
.block-container {{padding-top: 1.0rem; padding-bottom: 1rem; max-width: 100%;}}
.topbar {{display:flex; align-items:baseline; gap:14px; flex-wrap:wrap; margin-bottom:6px;}}
.topbar .title {{font-size:1.4rem; font-weight:700; color:#e6edf3;}}
.topbar .meta {{color:{GRAY}; font-size:0.9rem;}}
.chip {{display:inline-block; padding:2px 9px; border-radius:999px; font-size:0.76rem; font-weight:600;
       border:1px solid rgba(255,255,255,0.12); background:#1c2430; color:#c9d1d9; margin-right:6px;}}
.note {{color:{GRAY}; font-size:0.78rem;}}
.dist {{border:1px solid #2f3a48; border-left:4px solid {GRAY}; background:#15181d; border-radius:8px;
        padding:8px 12px; margin:4px 0 8px 0; font-size:0.92rem;}}
.jury {{border:1px solid #2f3a48; border-left:4px solid {BLUE}; background:#121820; border-radius:8px;
       padding:9px 14px; margin:4px 0 10px 0; font-size:0.92rem;}}
.jury b {{color:#e6edf3;}} .jury .warn {{color:{RED}; font-weight:600;}}
.pcard {{border:1px solid #232c38; background:#121820; border-radius:9px; padding:9px 11px; line-height:1.35;
         margin-bottom:6px;}}
.pcard.sel {{border-color:#e6edf3; background:#1a212b;}}
.pcard.flash {{animation: pflash 2.2s ease-out 1;}}
@keyframes pflash {{0% {{box-shadow:0 0 0 3px {YELLOW}; border-color:{YELLOW};}} 100% {{box-shadow:0 0 0 0 rgba(0,0,0,0);}}}}
table.cmp {{width:100%; border-collapse:collapse; font-size:0.9rem; color:#e6edf3;}}
table.cmp th {{text-align:left; color:#8b949e; font-weight:600; border-bottom:1px solid #2f3a48; padding:6px 8px;}}
table.cmp td {{border-bottom:1px solid #1c2430; padding:6px 8px; vertical-align:top;}}
table.cmp tr.grp td {{color:#8b949e; font-size:0.75rem; text-transform:uppercase; letter-spacing:0.06em;
                      padding-top:12px; border-bottom:none;}}
.fltsum {{border-left:3px solid {BLUE}; background:#121820; color:#e6edf3; border-radius:6px; padding:5px 10px;
          font-size:0.86rem; margin:2px 0 8px 0;}}
.pickinfo {{border-left:3px solid {YELLOW}; background:#1a1d12; color:#e6edf3; border-radius:6px; padding:5px 10px;
           font-size:0.84rem; margin:0 0 6px 0;}}
.pcard .name {{font-weight:700; font-size:0.95rem; color:#e6edf3;}}
.pcard .sub {{color:{GRAY}; font-size:0.78rem; margin-top:3px;}} .pcard .sub b {{color:#e6edf3;}}
.lcw {{color:{ORANGE}; font-size:0.74rem; font-weight:600; margin-left:6px;}}
.rsn {{border-bottom:1px solid #1c2430; padding:6px 0 5px 0;}}
.rsn .row {{display:flex; justify-content:space-between; align-items:baseline; gap:8px;}}
.rsn .n {{font-size:0.9rem; font-weight:600; color:#e6edf3;}}
.rsn .val {{font-size:0.98rem; font-weight:700; color:#e6edf3; white-space:nowrap;}}
.rsn .c {{font-size:0.76rem;}}
.lowdata {{color:{ORANGE}; font-weight:600;}}
.badge {{display:inline-block; padding:1px 7px; border-radius:6px; font-size:0.68rem; font-weight:700; color:#0b0f14;}}
.kv {{display:grid; grid-template-columns: 1fr auto; gap:3px 10px; font-size:0.84rem; margin:4px 0 6px 0;}}
.kv .k {{color:{GRAY};}} .kv .v {{color:#e6edf3; font-weight:600; text-align:right;}}
.sec {{font-size:0.7rem; letter-spacing:0.08em; text-transform:uppercase; color:{GRAY}; margin:8px 0 4px 0; font-weight:700;}}
.big {{font-size:1.2rem; font-weight:700; color:#e6edf3;}}
.lbl {{border:1px dashed #3a4350; border-radius:8px; padding:6px 10px; font-size:0.78rem; color:#c9d1d9; margin:6px 0;}}
.mgrid {{display:grid; grid-template-columns:1fr; gap:6px; margin:2px 0 4px 0;}}
.mcard {{border:1px solid #232c38; background:#121820; border-radius:8px; padding:6px 9px;}}
.mcard .g {{font-size:0.64rem; letter-spacing:0.06em; text-transform:uppercase; color:{GRAY};}}
.mcard .row {{display:flex; justify-content:space-between; align-items:baseline; gap:8px;}}
.mcard .n {{font-size:0.82rem; color:#c9d1d9;}} .mcard .val {{font-size:1.02rem; font-weight:700; color:#e6edf3;}}
.mcard .c {{font-size:0.74rem; color:{GRAY};}}
/* tıklanabilir oyuncu kartı: görünmez düğme kartın tamamını kaplar */
[class*="st-key-pc_"] {{position:relative; gap:0 !important;}}
[class*="st-key-pc_"] [data-testid="stElementContainer"]:has([data-testid="stButton"]) {{position:absolute; inset:0; z-index:3;}}
[class*="st-key-pc_"] [data-testid="stButton"], [class*="st-key-pc_"] [data-testid="stButton"] button {{width:100%; height:100%;}}
[class*="st-key-pc_"] [data-testid="stButton"] button {{opacity:0; cursor:pointer;}}
[class*="st-key-pc_"]:hover .pcard {{border-color:#4b5563; background:#161d26;}}
.hist-h {{color:{GRAY}; font-size:0.78rem; text-transform:uppercase; letter-spacing:0.05em;}}
.hist-c {{font-size:0.95rem; color:#e6edf3; padding-top:6px;}}
.hist-c .muted {{color:{GRAY}; font-size:0.85rem;}}
.cnt {{display:inline-block; min-width:26px; text-align:center; border-radius:6px; padding:1px 6px; margin-right:4px;
       font-weight:700; font-size:0.85rem; color:#0b0f14;}}
.step {{display:flex; align-items:center; gap:10px; padding:5px 0; font-size:1.0rem;}}
.step .dot {{width:22px; height:22px; border-radius:50%; display:inline-flex; align-items:center; justify-content:center;
             font-size:0.8rem; font-weight:700; color:#0b0f14;}}
.step.cur {{font-weight:700;}}
.jobhead {{font-size:1.15rem; font-weight:700; color:#e6edf3;}}
.jobtime {{font-size:1.0rem; color:#e6edf3;}}
</style>""", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def entries():
    root = os.environ.get("ANALYSIS_ROOT")     # verilmezse reports/analysis/ + samples/analysis/ (örnek paket)
    return V.list_analyses(Path(root) if root else None)


@st.cache_data(show_spinner=False, max_entries=8)
def playback_fig(work: str, rnd: int, hz: int, trail_s: float, height: int = 730, level: str | None = None):
    return PB.build_playback_figure(Path(work), rnd, hz, trail_s, height=height, level=level)


@st.cache_data(show_spinner=False)
def near_thresholds() -> dict:
    return V.aim_near_thresholds()


@st.cache_data(show_spinner=False)
def load(demo_id: str):
    # önbellek anahtarı demo_id: liste sırası yeni analiz eklenince kayar, sıra numarası eski demoyu açardı
    return V.load(next(e for e in entries() if e["demo_id"] == demo_id))


@st.cache_data(show_spinner=False)
def round_ctx(work: str, rnd: int):
    return V.context_round(Path(work), rnd)


@st.cache_data(show_spinner=False)
def radar_image(name: str):
    return Image.open(R.RADAR_DIR / name)


def esc(s) -> str:
    return html.escape(str(s))


def kv(rows) -> str:
    return '<div class="kv">' + "".join(f'<div class="k">{esc(k)}</div><div class="v">{esc(v)}</div>'
                                        for k, v in rows) + "</div>"


def badge(priority) -> str:
    p = priority or "—"
    return (f'<span class="badge" style="background:{PRIO_COLOR.get(p, "#6e7681")}">'
            f'{esc(p.replace(" inceleme önceliği", "").upper())}</span>')


def flag(v) -> str:
    """evet/hayır/bilinmiyor + simge."""
    t = V.yes_no(v)
    return {"evet": "● evet", "hayır": "○ hayır"}.get(t, "? bilinmiyor")


def metric_html(cards) -> str:
    def color(c):
        return pct_color(c["pct"], c["typical"])
    return '<div class="mgrid">' + "".join(
        f'<div class="mcard"><div class="g">{esc(c["group"])}</div><div class="row"><span class="n">{esc(c["name"])}</span>'
        f'<span class="val">{esc(c["value"])}</span></div><div class="c" style="color:{color(c)}">{esc(c["compare"])}'
        f'{" · tipik aralıkta" if c["typical"] else ""}</div></div>' for c in cards) + "</div>"


def pct_color(pct, typical) -> str:
    if typical or pct is None:
        return GRAY
    return RED if pct >= 95 else ORANGE if pct >= 90 else "#c9d1d9"


def reasons_html(cards) -> str:
    """en önemli 3 kısa neden (ad · değer · karşılaştırma). Ayrıntılı kartlar 'Detaylı metrikler'de."""
    return '<div class="rsns">' + "".join(
        f'<div class="rsn"><div class="row"><span class="n">{esc(r["name"])}</span><span class="val">{esc(r["value"])}'
        f'</span></div><div class="c" style="color:{pct_color(r["pct"], r["typical"])}">{esc(r["compare"])}</div></div>'
        for r in V.reason_lines(cards)) + "</div>"


# ---------------------------------------------------------------- .dem yükleme: form + ilerleme paneli
def upload_form():
    """Kenar çubuğu: .dem seç → güvenli kaydet (içerik hash'i) → daha önce analiz edildiyse aç, değilse başlat."""
    busy = U.running_job()
    with st.sidebar.expander("Yeni demo analizi", expanded=bool(st.session_state.get("job")) or not E):
        up = st.file_uploader(".dem dosyası (en fazla ~1 GB)", type=["dem"], key="dem_upload")
        source = st.selectbox("Kaynak", U.SOURCES, key="dem_source",
                              help="valve = Premier/Rekabetçi (Valve matchmaking). Ray-cast kontrol ölçütünü etkiler.")
        keep = st.checkbox("Ham demo dosyasını analizden sonra sakla", value=False, key="dem_keep",
                           help="Varsayılan: kapalı. Analiz başarıyla bitince ham .dem silinir. Gerçek kimlik ↔ "
                                "Player_N eşlemesi her durumda silinir; yalnız anonim analiz çıktıları kalır.")
        if busy:
            st.caption("Şu anda bir analiz çalışıyor; aynı anda tek analiz yapılır.")
        if st.button("Analizi başlat", type="primary", disabled=up is None or bool(busy), key="dem_start",
                     width="stretch"):
            try:
                did, _ = U.save_upload(up, up.name, up.size)
            except U.UploadError as ex:
                st.error(str(ex))
                return
            if U.is_done(did):
                st.session_state.pop("job", None)
                entries.clear()
                st.query_params["demo"] = did
                st.toast("Bu demo daha önce analiz edilmiş; mevcut sonuç açılıyor.")
                st.rerun()
            if U.running_job() == did or U.start_job(did, source, keep_raw=keep):
                st.session_state["job"] = did
                st.rerun()
            st.error("Başka bir analiz çalışıyor. Bitince yeniden deneyin.")
        st.caption("Dosyanın adı hiçbir yere yazılmaz; kimlik = içerik hash'i. Analiz ayrı süreçte çalışır "
                   "(görünürlük adımı haritaya göre ~2–8 dk).")


def stage_html(did: str, job: dict | None) -> str:
    """7 adımlı gösterim: mevcut aşama işaretlerinden okunur; yüzde/kalan süre yok."""
    pv = U.progress_view(did, job)
    rows = []
    for i, stp in enumerate(pv["steps"]):
        icon, col = {"done": ("✓", GREEN), "running": (str(i + 1), YELLOW), "failed": ("✕", RED),
                     "pending": (str(i + 1), "#3a4553")}[stp["state"]]
        cls = "step cur" if stp["state"] in ("running", "failed") else "step"
        txt = "#e6edf3" if stp["state"] != "pending" else GRAY
        rows.append(f'<div class="{cls}"><span class="dot" style="background:{col}">{icon}</span>'
                    f'<span style="color:{txt}">{esc(stp["label"])}</span></div>')
    return "".join(rows)


def _mmss(sec: float) -> str:
    sec = int(sec or 0)
    return f"{sec // 60} dk {sec % 60:02d} sn"


@st.fragment(run_every=2)
def job_progress(did: str):
    """Çalışan analizin canlı ilerlemesi (2 sn'de bir). Bitince sonuç ekranına otomatik geçer.
    Gösterilenler ölçülmüş bilgiler: mevcut aşama, toplam süre, bu aşamada geçen süre; yüzde/tahmini süre YOK."""
    job = U.read_job(did) or {}
    if job.get("state") == "done":
        st.session_state.pop("job", None)
        entries.clear()
        st.query_params["demo"] = did
        st.rerun(scope="app")
    if job.get("state") == "failed":
        st.rerun(scope="app")
    pv = U.progress_view(did, job)
    st.markdown(f'<div class="jobhead">Analiz sürüyor · demo_id {esc(did)}</div>'
                f'<div class="jobtime">Aşama {pv["index"] + 1 if pv["index"] is not None else pv["n"]} / {pv["n"]}: '
                f'<b>{esc(pv["current"] or "Tamamlandı")}</b> · bu aşamada {_mmss(pv["elapsed_stage"])} · '
                f'toplam {_mmss(pv["elapsed_total"])}</div>', unsafe_allow_html=True)
    st.markdown(stage_html(did, job), unsafe_allow_html=True)
    if pv["current_key"] == "gorunurluk":
        txt = U.past_stage_text("gorunurluk")
        if txt:
            st.markdown(f'<div class="fltsum">{esc(txt)}</div>', unsafe_allow_html=True)
    st.caption("Sayfayı kapatsanız da analiz arka planda sürer; bitince bu demo listede görünür.")


def job_panel():
    did = st.session_state.get("job") or U.running_job()
    if not did:
        return
    job = U.read_job(did) or {}
    with st.container(border=True):
        if job.get("state") == "failed":
            st.error(f"Analiz tamamlanamadı: {job.get('message')}")
            st.markdown(stage_html(did, job), unsafe_allow_html=True)
            st.caption("Ham demo ve ara dosyalar SİLİNMEDİ (hata ayıklama / kaldığı yerden devam için).")
            if job.get("log_tail") or job.get("problems"):
                with st.expander("Teknik ayrıntı"):
                    st.code(job.get("log_tail") or "\n".join(job.get("problems") or []), language="text")
            b1, b2, b3 = st.columns(3)
            if b1.button("Kaldığı yerden yeniden dene", key="job_retry", disabled=bool(U.running_job())):
                if not (U.UPLOAD_DIR / f"{did}.dem").exists():
                    st.error("Ham demo bulunamadı; dosyayı yeniden yükleyin.")
                elif U.start_job(did, job.get("source", "other"), keep_raw=job.get("keep_raw", False), resume=True):
                    st.rerun()
            if b2.button("Ham demoyu ve geçici dosyaları sil", key="job_discard"):
                U.discard(did)
                st.session_state.pop("job", None)
                st.rerun()
            if b3.button("Kapat (dosyalar kalsın)", key="job_close"):
                st.session_state.pop("job", None)
                st.rerun()
        elif job.get("state") == "running" or U.running_job() == did:
            job_progress(did)
        else:
            st.session_state.pop("job", None)


# ---------------------------------------------------------------- sunum modu (yalnız CSS/gösterim)
PRESENT = st.sidebar.toggle("Sunum modu", key="present_mode",
                            help="Projeksiyon için: yazılar ve rozetler büyür, ikincil notlar gizlenir, radar büyür. "
                                 "Skor, seçim, filtre ve analiz mantığı değişmez. Sunumda kenar çubuğunu sol üstteki "
                                 "« ile kapatarak radara daha fazla yer açın.")
if PRESENT:
    st.markdown(f"""<style>
html {{font-size: 17px !important;}}
@media (max-width: 1500px) {{ html {{font-size: 16px !important;}} }}
.pcard > div:first-child {{flex-wrap: wrap; gap: 2px 6px;}}
.badge {{font-size: 0.74rem !important; padding: 2px 9px !important;}}
.pcard .name {{font-size: 1.08rem !important;}} .pcard .sub {{font-size: 0.82rem !important;}}
.sec {{font-size: 0.85rem !important;}}
[data-testid="stMain"] [data-testid="stCaptionContainer"] {{display: none !important;}}
[data-testid="stMain"] .note {{display: none !important;}}
[data-testid="stMain"] .dist .note, [data-testid="stMain"] .jury .note, [data-testid="stMain"] .fltsum .note
    {{display: inline !important;}}
.chip {{font-size: 0.85rem !important;}}
/* seçili oyuncu, rozet ve 3 neden baskın; katlanır bölüm başlıkları geri planda */
.selp .big {{font-size: 1.5rem !important;}} .selp .badge {{font-size: 0.9rem !important; padding: 3px 11px !important;}}
.pcard.sel {{border-width: 2px !important;}}
.rsn .n {{font-size: 1.0rem !important;}} .rsn .val {{font-size: 1.1rem !important;}} .rsn .c {{font-size: 0.82rem !important;}}
[data-testid="stMain"] [data-testid="stExpander"] summary {{opacity: 0.6; font-size: 0.85rem;}}
@media (max-width: 1500px) {{ .selp .big {{font-size: 1.3rem !important;}} .rsn .n {{font-size: 0.9rem !important;}}
    .rsn .val {{font-size: 1.0rem !important;}} }}
</style>""", unsafe_allow_html=True)
RADAR_H = 860 if PRESENT else 730
COLS = [0.9, 3.6, 1.2] if PRESENT else [0.85, 3.4, 1.2]

# ---------------------------------------------------------------- kenar çubuğu: demo seçimi
E = entries()
st.sidebar.markdown("### Analizler")
if E:
    _want = st.query_params.get("demo")     # ?demo=<demo_id> ile doğrudan açılır (örnek bağlantısı, yükleme sonrası)
    _default = V.default_index(E, _want)
    idx = st.sidebar.selectbox("Önceden analiz edilmiş demolar", range(len(E)), index=_default,
                               format_func=lambda i: E[i]["title"], key=f"demo_sel_{len(E)}_{_want}",
                               on_change=lambda: st.session_state.update(show_history=False))
    st.sidebar.button("Analiz geçmişi", key="hist_open", width="stretch",
                      on_click=lambda: st.session_state.update(show_history=True))
upload_form()
job_panel()
if not E:
    st.info("Henüz analiz yok. Soldaki 'Yeni demo analizi' ile bir .dem yükleyin.")
    st.stop()


def _open_from_history(did: str):
    """'Aç': yalnız görünüm — demoyu açar, inceleme ekranına döner."""
    st.query_params["demo"] = did
    st.session_state["show_history"] = False


if st.session_state.get("show_history"):
    # analiz geçmişi (yalnız mevcut dosyalar; yeni hesap yok)
    st.markdown('<div class="topbar"><span class="title">Analiz geçmişi</span>'
                f'<span class="meta">{len(E)} analiz · yeniden eskiye</span></div>', unsafe_allow_html=True)
    st.caption("Tarih = analiz dosyasının tarihi. Örnek paketteki analizlerde tarih gösterilmez ('örnek paket'). "
               "Sayılar mevcut inceleme önceliği çıktılarıdır.")
    W = [1.25, 1.2, 1.0, 0.55, 1.15, 1.7, 1.25, 2.1, 0.55]
    hc = st.columns(W)
    for c, h in zip(hc, ["demo_id", "kaynak", "harita", "round", "analiz tarihi", "yüksek / orta / düşük / yetersiz",
                         "dağılım", "rol", ""]):
        c.markdown(f'<div class="hist-h">{h}</div>', unsafe_allow_html=True)
    for r in V.history_rows(E):
        cs = st.columns(W, vertical_alignment="center")
        cnt = "".join(f'<span class="cnt" style="background:{col}">{n}</span>' for n, col in
                      ((r["high"], RED), (r["mid"], ORANGE), (r["low"], GREEN), (r["insufficient"], GRAY)))
        vals = [f"<b>{esc(r['demo_id'])}</b>", esc(r["source"]), esc(r["map"]), esc(r["rounds"]),
                f'<span class="muted">{esc(r["date"])}</span>' if r["timestamp"] is None else esc(r["date"]), cnt,
                esc(r["dist"]), f'<span class="muted">{esc(r["role"]) or "—"}</span>']
        for c, v in zip(cs[:-1], vals):
            c.markdown(f'<div class="hist-c">{v}</div>', unsafe_allow_html=True)
        cs[-1].button("Aç", key=f"hist_{r['demo_id']}", on_click=_open_from_history, args=(r["demo_id"],),
                      width="stretch")
    st.stop()
d = load(E[idx]["demo_id"])
entry, meta, info = d["entry"], d["meta"], d["radar"]
st.sidebar.caption(f"demo_id: {meta['demo_id']}")

# ---------------------------------------------------------------- üst bant
rt = V.live_rounds(meta)
src_txt = "CS2CD · eğitim havuzu (OOF)" if entry["mode"] == "cs2cd_oof" else f"güncel demo · {meta.get('source')}"
st.markdown(f'<div class="topbar"><span class="title">CS2 Davranış Analizi</span>'
            f'<span class="meta">{esc(meta["map"])} · {meta.get("rounds")} round · {esc(src_txt)}'
            + (f' · {esc(entry["match"])}' if entry.get("match") else "") + "</span></div>", unsafe_allow_html=True)
chips = f'<span class="chip">{esc(S.DISCLAIMER)}</span><span class="chip">Olay skorları = inceleme noktası, kanıt değil</span>'
if entry["mode"] == "cs2cd_oof":
    chips += f'<span class="chip" style="border-color:{BLUE}">{esc(d["result"]["mode_note"])}</span>'
st.markdown(chips, unsafe_allow_html=True)

jury = entry.get("jury") or {}
focus_players = [p for p in (jury.get("flagged_players") or [])]
if jury:
    title, body = JURY_TEXT.get(jury["role"], (jury["label"], ""))
    extra = f"<br>Bakılacak oyuncu: <b>{esc(', '.join(focus_players))}</b> (açılışta seçili)." if focus_players else ""
    if jury["role"] == "yanlis_alarm":
        extra += f' <span class="warn">{esc(SNIPER_LIMIT)}</span>'
    st.markdown(f'<div class="jury"><b>Bu örnek neyi gösteriyor? — {esc(title)}</b><br>{esc(body)}{extra}'
                f'<br><span class="note">Örnek analiz: CS2CD veri setinden önceden belirlenmiş bir kuralla seçildi (seed 2026).</span></div>',
                unsafe_allow_html=True)

dist = V.distribution(d["work"])     # demo düzeyi, yalnız bilgi (skorları değiştirmez)
if dist and dist["level"] > 0:
    import distribution_check as DC
    why = DC.reasons(dist)
    st.markdown(f'<div class="dist"><b>Demo düzeyi bilgi — {esc(dist["title"])}</b><br>{esc(dist["text"])}'
                f'<br>Neden: {esc(why)}.<br><span class="note">{esc(dist["note"])} Ayrıntı: "Model sınırlılıkları".'
                '</span></div>', unsafe_allow_html=True)

if info is None:
    st.error(f"{meta['map']} için konum verisi yok; 2D görünüm çizilemiyor.")
    st.stop()

# ---------------------------------------------------------------- seçim durumu
P = d["players"]
order = V.player_order(P)
attackers = set(d["events"]["attacker"].unique().to_list())
key_p = f"player_{meta['demo_id']}"
key_pick, key_nonce = f"tl_pick_{meta['demo_id']}", f"tl_nonce_{meta['demo_id']}"
key_mode = f"mode_{meta['demo_id']}"
picked, flash_player = st.session_state.pop(key_pick, None), None
if picked and picked["player"] in order:            # zaman çizelgesinden tıklama (timeline_pick, kill_id tabanlı)
    prev = st.session_state.get(key_p)
    st.session_state[key_p] = picked["player"]
    st.session_state[f"ev_{meta['demo_id']}_{picked['player']}"] = picked["kill_id"]
    st.session_state[key_mode] = MODE_EVENT          # oynatmadayken tıklandıysa olay incelemeye dön (yalnız görünüm)
    if prev and prev != picked["player"]:
        flash_player = picked["player"]
        st.toast(f"Oyuncu değişti: {prev} → {picked['player']} (zaman çizelgesinde seçilen olayın saldırganı)", icon="🎯")
else:
    picked = None
nonce = st.session_state.get(key_nonce, 0)
if key_p not in st.session_state or st.session_state[key_p] not in order:
    # örnek kutusu bir oyuncuyu gösteriyorsa sayfa o oyuncuyla açılır; değilse listenin ilki
    st.session_state[key_p] = next((p for p in focus_players + order if p in attackers and p in order), order[0])
player = st.session_state[key_p]
prow = P.filter(pl.col("player") == player).row(0, named=True)
labels = dict(d["labels"].iter_rows()) if d["labels"] is not None else {}

# ---------------------------------------------------------------- HTML inceleme raporu (yalnız mevcut çıktılar)
with st.sidebar.expander("İnceleme raporu (HTML)"):
    _rk = f"rep_html_{meta['demo_id']}"
    _cmp_on = st.checkbox("Oyuncu karşılaştırmasını ekle", key=f"rep_cmp_{meta['demo_id']}",
                          help="A/B: karşılaştırma ekranındaki seçim; yoksa seçili oyuncu ve öncelik sırasındaki sonraki.")
    if st.button("Rapor oluştur", key=f"rep_make_{meta['demo_id']}", width="stretch"):
        import report_export as RE
        _a = st.session_state.get(f"cmp_a_{meta['demo_id']}", player)
        _b = st.session_state.get(f"cmp_b_{meta['demo_id']}", next((x for x in order if x != player), player))
        try:
            st.session_state[_rk] = RE.build_report(d["work"], (_a, _b) if _cmp_on else None)
        except RE.ReportPrivacyError as _err:
            st.session_state.pop(_rk, None)
            st.error(f"Rapor gizlilik denetiminden geçmedi, oluşturulmadı: {_err}")
    if st.session_state.get(_rk):
        st.download_button("HTML raporu indir", data=st.session_state[_rk].encode("utf-8"),
                           file_name=f"inceleme_raporu_{meta['demo_id']}.html", mime="text/html",
                           key=f"rep_dl_{meta['demo_id']}", width="stretch")
        st.caption(f"{len(st.session_state[_rk].encode('utf-8')) / 1e3:.0f} KB · tek dosya. PDF için tarayıcıda açıp "
                   "Ctrl+P → PDF olarak kaydet. Rapor mevcut çıktılardan üretilir; yeni skor/eşik uygulanmaz.")

# ---------------------------------------------------------------- görünüm: İnceleme / Oyuncu karşılaştırma
# Karşılaştırma YALNIZ GÖSTERİM: mevcut model çıktıları + özellik özetleri; yeni skor, eşik, sıralama, hüküm yok.
key_view = f"view_{meta['demo_id']}"
if key_view not in st.session_state:
    st.session_state[key_view] = VIEW_INSPECT


def _open_in_inspect(p: str):
    """'İncelemede aç': yalnız görünümü değiştirir ve oyuncuyu inceleme ekranında seçer."""
    st.session_state[key_p] = p
    st.session_state[key_view] = VIEW_INSPECT


view = st.segmented_control("Görünüm", [VIEW_INSPECT, VIEW_COMPARE], key=key_view,
                            label_visibility="collapsed") or VIEW_INSPECT
if view == VIEW_COMPARE:
    # inceleme ekranının widget'ları bu görünümde çizilmez; Streamlit durumlarını silmesin (filtre, olay, radar modu)
    for k in [k for k in list(st.session_state.keys()) if isinstance(k, str) and k.startswith(
            (f"flt_{meta['demo_id']}", f"fltp_{meta['demo_id']}", f"ev_{meta['demo_id']}_", key_mode))]:
        st.session_state[k] = st.session_state[k]
    opts = order
    b_default = next((p for p in order if p != player), player)
    ca, cb = st.columns(2)
    a = ca.selectbox("Oyuncu A", opts, index=opts.index(player), key=f"cmp_a_{meta['demo_id']}")
    b = cb.selectbox("Oyuncu B", opts, index=opts.index(b_default), key=f"cmp_b_{meta['demo_id']}")
    ca.button(f"{a} — İncelemede aç", key=f"cmp_open_a_{meta['demo_id']}", on_click=_open_in_inspect, args=(a,))
    cb.button(f"{b} — İncelemede aç", key=f"cmp_open_b_{meta['demo_id']}", on_click=_open_in_inspect, args=(b,))
    rows = V.compare_players(d, a, b)
    st.markdown(f'<div class="fltsum">{esc(V.COMPARE_NOTE)}</div>', unsafe_allow_html=True)
    sty = {r["key"]: r for r in rows}["style"]
    if sty["a"] and sty["b"] and sty["a"] != sty["b"]:
        st.caption(f"Oyuncuların stili farklı (A: {sty['a']}, B: {sty['b']}). Yüzdelikler her oyuncunun kendi "
                   "stilindeki temiz referansa göredir; grafik yalnız konumu gösterir, iki oyuncuyu birbirine göre "
                   "yorumlamaz.")
    for p, lab in ((a, "A"), (b, "B")):
        k = P.filter(pl.col("player") == p)["analyzed_kills"]
        k = int(k[0]) if k.len() and k[0] is not None else 0
        if k < V.LOW_DATA_KILLS:
            st.caption(f"{lab} ({p}): {k} öldürme — " + ("yetersiz veri (< 3 öldürme), model skoru yok."
                                                        if k < 3 else "az veri; ortalamalar daha oynak."))

    def cell(r, side):
        txt = r[f"{side}_text"]
        if r["key"] == "review_priority":
            return badge(r[side])
        pct = r.get(f"{side}_pct")
        extra = (f'<div class="note">%{pct:.0f} yüzdelik</div>' if pct is not None and r["key"] != "style_percentile"
                 else "")
        n = r.get(f"{side}_n")
        nn = f'<div class="note">{n} öldürmeden</div>' if r["kind"] == "feature" and n is not None else ""
        return f"{esc(txt)}{extra}{nn}"

    def diff(r):
        dname = r["direction"]
        col = CMP_A if dname == "A daha yüksek" else CMP_B if dname == "B daha yüksek" else GRAY
        return f'<span style="color:{col};font-weight:600">{esc(dname)}</span>'

    html_rows, grp = [], None
    for r in rows:
        if r["group"] != grp:
            grp = r["group"]
            html_rows.append(f'<tr class="grp"><td colspan="4">{esc(grp)}</td></tr>')
        note = f'<div class="note">{esc(r["note"])}</div>' if r.get("note") else ""
        html_rows.append(f'<tr><td>{esc(r["label"])}{note}</td><td>{cell(r, "a")}</td><td>{cell(r, "b")}</td>'
                         f'<td>{diff(r)}</td></tr>')
    left_c, right_c = st.columns([1.35, 1], gap="medium")
    left_c.markdown(
        f'<table class="cmp"><tr><th>metrik</th><th style="color:{CMP_A}">A · {esc(a)}</th>'
        f'<th style="color:{CMP_B}">B · {esc(b)}</th><th>yön</th></tr>' + "".join(html_rows) + "</table>",
        unsafe_allow_html=True)
    left_c.caption("Yüzdelik = aynı stildeki temiz eğitim oyuncularının oyuncu ortalamalarının yüzde kaçı bu değerden "
                   "küçük (0 = en düşük, 100 = en yüksek). 'Yön' yalnız hangi değerin sayıca büyük olduğunu söyler; "
                   "iyi/kötü ya da hile anlamı taşımaz.")
    with left_c.expander("Ek nişan özellikleri"):
        extra_rows = [r for r in V.compare_players(d, a, b, extra=True) if r["group"] == "Nişan (ek)"]
        st.markdown('<table class="cmp">' + "".join(
            f'<tr><td>{esc(r["label"])}</td><td>{cell(r, "a")}</td><td>{cell(r, "b")}</td><td>{diff(r)}</td></tr>'
            for r in extra_rows) + "</table>", unsafe_allow_html=True)
    # nokta grafiği: temiz referanstaki konum (yalnız konum; yorum yok)
    pr_ = [r for r in rows if r.get("a_pct") is not None or r.get("b_pct") is not None]
    labels_ = [r["label"] if r["key"] != "style_percentile" else "Model skoru (stil içi yüzdelik)" for r in pr_][::-1]
    pr_ = pr_[::-1]
    cf = go.Figure()
    for i, r in enumerate(pr_):
        xs = [v for v in (r.get("a_pct"), r.get("b_pct")) if v is not None]
        if len(xs) == 2:
            cf.add_trace(go.Scatter(x=xs, y=[labels_[i]] * 2, mode="lines", line=dict(color="#3a4553", width=3),
                                    hoverinfo="skip", showlegend=False))
    for side, col, sym, nm in (("a", CMP_A, "circle", f"A · {a}"), ("b", CMP_B, "diamond", f"B · {b}")):
        cf.add_trace(go.Scatter(x=[r.get(f"{side}_pct") for r in pr_], y=labels_, mode="markers", name=nm,
                                marker=dict(size=13, color=col, symbol=sym, line=dict(color="#0b0f14", width=1)),
                                hovertemplate="%{y}: %{x:.0f}. yüzdelik<extra>" + esc(nm) + "</extra>"))
    cf.add_vline(x=50, line=dict(color="#3a4553", width=1, dash="dot"))
    cf.update_xaxes(range=[-2, 102], title=dict(text="temiz referansta yüzdelik (0–100)", font=dict(size=11, color=GRAY)),
                    gridcolor="#1c2430", tickfont=dict(color=GRAY), zeroline=False)
    cf.update_yaxes(tickfont=dict(color="#c9d1d9", size=11), automargin=True)
    cf.update_layout(height=80 + 38 * len(pr_), margin=dict(l=10, r=10, t=30, b=10), plot_bgcolor="#0b0f14",
                     paper_bgcolor="#0b0f14", legend=dict(orientation="h", x=0, y=1.08, font=dict(color="#c9d1d9")))
    right_c.markdown('<div class="sec">Temiz referansa göre konum</div>', unsafe_allow_html=True)
    right_c.plotly_chart(cf, width="stretch", config={"displaylogo": False}, key=f"cmp_fig_{meta['demo_id']}")
    right_c.caption("Grafik her oyuncunun kendi stilindeki temiz referansa göre konumunu gösterir; oyuncuları "
                    "birbirine göre değerlendirmez.")
    st.stop()


# ---------------------------------------------------------------- olay filtreleri (yalnız gösterim/inceleme)
# Zaman çizelgesi, olay listesi ve radar AYNI filtreli kümeyi (fe) kullanır; skor/öncelik/eşik değişmez.
key_f, key_fp = f"flt_{meta['demo_id']}", f"fltp_{meta['demo_id']}"


def _clear_filters():
    st.session_state[key_f] = []
    st.session_state[key_fp] = 0


# başlık SABİT: değişirse Streamlit bölümü yeni öğe sayıp kapatır (filtre seçince panel kapanıyordu; final regresyonda
# bulundu). Aktif filtre özeti bölümün dışında ayrıca gösterilir.
fbox = st.expander("Olay filtreleri", expanded=False)
fc1, fc3 = fbox.columns([5.2, 0.8], vertical_alignment="bottom")     # filtre butonları tam satır (kesilmesin)
active = fc1.pills("Olay filtreleri (seçilenlerin hepsi birlikte: VE)", list(V.FILTERS), selection_mode="multi",
                   key=key_f, format_func=lambda k: V.FILTERS[k][0],
                   help="\n\n".join(f"**{lab}:** {txt}" for lab, txt in V.FILTERS.values())) or []
fc2, fsum = fbox.columns([1.4, 4.6], vertical_alignment="center")
min_pct = fc2.select_slider("Yüksek olay skoru (stil içi olay yüzdeliği)", V.SCORE_PCT_STEPS, key=key_fp,
                            format_func=lambda v: "kapalı" if v == 0 else f"≥ %{v}")
filters_on = bool(active) or bool(min_pct)
fc3.button("Filtreleri temizle", key=f"fltc_{meta['demo_id']}", on_click=_clear_filters, disabled=not filters_on,
           width="stretch")
fe = V.filter_events(d["events"], active, min_pct, near_thresholds())
if filters_on:
    notes = [f"<b>{esc(V.FILTERS[k][0])}:</b> {esc(V.FILTERS[k][1])}" for k in ("not_visible", "aim_near_hidden")
             if k in active]
    # aktif filtre özeti katlanır bölümün DIŞINDA görünür kalır (kapalıyken de ne süzüldüğü belli)
    st.markdown(f'<div class="fltsum">{esc(V.filter_summary(active, min_pct, fe.height, d["events"].height))}'
                + "".join(f'<br><span class="note">{n}</span>' for n in notes) + "</div>", unsafe_allow_html=True)
else:
    fsum.caption(f"Filtre yok · {d['events'].height} olayın hepsi gösteriliyor. Filtreler yalnız inceleme içindir; "
                 "skorları ve öncelikleri değiştirmez.")

left, center, right = st.columns(COLS, gap="small")

# ---------------------------------------------------------------- sol: oyuncu kartları (tamamı tıklanabilir)
with left:
    st.markdown('<div class="sec">Oyuncular · inceleme önceliği</div>', unsafe_allow_html=True)
    for p in order:
        r = P.filter(pl.col("player") == p).row(0, named=True)
        kills = r.get("analyzed_kills")
        low = kills is not None and kills < V.LOW_DATA_KILLS
        with st.container(key=f"pc_{p}"):
            st.markdown(
                # ad + rozet, altında tek özet satırı; stil üzerine gelince (title), işaretler yalnız gerekirse
                f'<div class="pcard{" sel" if p == player else ""}{" flash" if p == flash_player else ""}" '
                f'title="Stil: {esc(r.get("style") or V.UNKNOWN)}"><div style="display:flex;justify-content:space-between;gap:4px 6px;flex-wrap:wrap;'
                f'align-items:center"><span class="name">{esc(p)}</span>{badge(r.get("review_priority"))}</div>'
                f'<div class="sub"><b>{V.fmt_pct(r.get("style_percentile"))}</b> stil içi · '
                f'<b>{int(r.get("review_points") or 0)}</b> ★ · <b>{V.fmt_value(kills, 0)}</b> öldürme'
                + (" · ⚠" if r.get("low_confidence_style") else "")
                + (' · <span class="lowdata">az veri</span>' if low else "") + "</div></div>",
                unsafe_allow_html=True)
            if p in attackers and p != player:
                if st.button(f"{p} seç", key=f"pick_{p}", width="stretch"):
                    st.session_state[key_p] = p
                    st.rerun()
    st.caption("★ inceleme noktası · ⚠ sniper ağırlıklı (düşük güven) · az veri: < 10 öldürme · "
               "stil: kartın üzerine gelin")

# ---------------------------------------------------------------- sağ (önce: olay seçimi radarı belirler)
ev = fe.filter(pl.col("attacker") == player).sort("event_score", descending=True)
key_ev, dropped = f"ev_{meta['demo_id']}_{player}", None
if ev.height and key_ev in st.session_state and st.session_state[key_ev] not in ev["kill_id"].to_list():
    dropped = st.session_state[key_ev]                  # seçili olay filtre dışında kaldı → filtreli en yüksek skorlu
    st.session_state[key_ev] = ev["kill_id"][0]
with right:
    st.markdown('<div class="sec">Seçili oyuncu</div>', unsafe_allow_html=True)
    # ilk bakışta yalnız temel alanlar + 3 kısa neden; teknik ayrıntı ve uzun notlar "Detaylı metrikler"de
    st.markdown(f'<div class="selp" style="display:flex;justify-content:space-between;align-items:center">'
                f'<span class="big">{esc(player)}</span>{badge(prow.get("review_priority"))}</div>'
                + ('<div class="lcw" style="margin:2px 0 0 0">⚠ düşük güven (sniper ağırlıklı)</div>'
                   if prow.get("low_confidence_style") else ""), unsafe_allow_html=True)
    st.markdown(kv([("Model skoru (ortalama)", V.fmt_value(prow.get("player_raw_score"), 3)),
                    ("Stil içi yüzdelik", V.fmt_pct(prow.get("style_percentile"))),
                    ("İnceleme noktası", int(prow.get("review_points") or 0)),
                    ("Modele giren öldürme", V.fmt_value(prow.get("analyzed_kills"), 0))]),
                unsafe_allow_html=True)
    cards = V.metric_cards(prow.get("player_top_features"))
    if cards:
        st.markdown('<div class="sec">Neden bu öncelik?</div>' + reasons_html(cards), unsafe_allow_html=True)
    with st.expander("Detaylı metrikler"):
        st.markdown(kv([("Stil", prow.get("style") or V.UNKNOWN),
                        ("Oyuncu eşiği (yüksek)", "> " + V.fmt_value(prow.get("g1_threshold"), 3))]),
                    unsafe_allow_html=True)
        if prow.get("low_confidence_style"):
            st.caption("⚠ " + LOW_CONF)
        if cards:
            st.markdown(metric_html(cards), unsafe_allow_html=True)
            st.caption(f"Karşılaştırma: aynı stildeki ({prow.get('style')}) temiz oyuncuların oyuncu ortalamaları. "
                       "'Tipik aralıkta': değer temizlerin %75'inden daha uç değil.")
    if player in labels:
        st.markdown(f'<div class="lbl">{esc(labels[player])}<br><span class="note">Veri seti etiketi — '
                    f'model çıktısı DEĞİLDİR; model etiketi kullanmaz.</span></div>', unsafe_allow_html=True)

    st.markdown('<div class="sec" style="margin-top:12px">Seçili olay</div>', unsafe_allow_html=True)
    if picked:
        st.markdown(f'<div class="pickinfo">Zaman çizelgesinden seçildi: olay #{picked["kill_id"]} · R{picked["round"]}'
                    + (f' · oyuncu <b>{esc(picked["player"])}</b> olarak değişti' if flash_player else "") + "</div>",
                    unsafe_allow_html=True)
    if ev.height == 0:
        n_all = d["events"].filter(pl.col("attacker") == player).height
        if filters_on and n_all:
            st.info(f"{player} oyuncusunun bu filtrelere uyan olayı yok (0 / {n_all}). Zaman çizelgesinden başka bir "
                    "olay seçin veya filtreleri gevşetin.")
        else:
            st.caption("Bu oyuncunun modele giren öldürmesi yok.")
        e = None
    else:
        if dropped is not None:
            st.caption(f"Önceki olay (#{dropped}) filtre dışında kaldı; filtreye uyan en yüksek skorlu olay seçildi.")
        kill_id = st.selectbox("Olay (olay skoruna göre)" + (" · filtreli" if filters_on else ""), ev["kill_id"].to_list(),
                               key=key_ev,
                               format_func=lambda k: (lambda r: f"{'★ ' if r['review_point'] else ''}#{k} · R{r['round']} · "
                                                      f"{r['weapon']} · {r['event_score']:.3f}")(
                                   ev.filter(pl.col("kill_id") == k).row(0, named=True)))
        e = ev.filter(pl.col("kill_id") == kill_id).row(0, named=True)
        thr = e.get("event_threshold")
        rp_txt = ("★ evet" if e["review_point"] else "hayır") + (
            f" ({e['event_score']:.3f} {'>' if e['review_point'] else '≤'} {thr:.3f})" if thr is not None else "")
        # önde temel bilgiler + 3 kısa neden; diğer teknik alanlar "Olay ayrıntıları"nda
        st.markdown(kv([("Round · tick", f"R{e['round']} · {e['kill_tick']}"),
                        ("Silah → kurban", e["weapon"] + (" (bıçak)" if e.get("is_knife") else "") + f" → {e['victim']}"),
                        ("Olay skoru", V.fmt_value(e["event_score"], 3)),
                        ("İnceleme noktası", "★ evet" if e["review_point"] else "hayır")]), unsafe_allow_html=True)
        ec = [] if e.get("is_knife") else V.metric_cards(e.get("top_features"))
        if ec:
            st.markdown('<div class="sec">Bu olayda öne çıkan</div>' + reasons_html(ec), unsafe_allow_html=True)
        with st.expander("Olay ayrıntıları"):
            st.markdown(kv([("Stil içi olay yüzdeliği", V.fmt_pct(e["event_style_percentile"])),
                            ("Olay eşiği (stil içi %95)", "> " + V.fmt_value(thr, 3)),
                            ("İnceleme noktası", rp_txt),
                            ("Wallbang", flag(None if e["wallbang"] is None else e["wallbang"] > 0)),
                            ("Duman içinden", flag(e["thrusmoke"])),
                            ("Görünürlük (ray-cast)", V.visibility_text(e["victim_visible_at_kill"])),
                            ("Tepki süresi", V.fmt_value(e["reaction_ms"], 0, " ms")),
                            ("Pre-FOV açısı", V.fmt_value(e["prefov_angle"], 1, "°"))]), unsafe_allow_html=True)
            if e.get("is_knife"):
                st.caption("Bıçak öldürmesi: v1'de 'otomatik' sınıfına düşer; özellik açıklaması üretilmez.")
            elif ec:
                st.markdown(metric_html(ec), unsafe_allow_html=True)
                if e.get("explanation_basis"):
                    st.caption(f"Karşılaştırma: {e['explanation_basis']} dağılımı.")
            st.markdown('<div class="sec">İnceleme noktası nedir?</div>', unsafe_allow_html=True)
            st.caption(POINT_DEF)

# ---------------------------------------------------------------- orta: radar
with center:
    if key_mode not in st.session_state:
        st.session_state[key_mode] = MODE_EVENT
    mode = st.segmented_control("Radar görünümü", [MODE_EVENT, MODE_PLAY], key=key_mode,
                                label_visibility="collapsed") or MODE_EVENT
    rounds = sorted(set(r for r in d["map_events"]["round"].unique().to_list() + d["events"]["round"].unique().to_list()
                        if r and r > 0))
    ev_round = e["round"] if e is not None else rounds[0]
    c1, c2 = st.columns([1, 5])
    rnd = c1.selectbox("Round", rounds, index=rounds.index(ev_round) if ev_round in rounds else 0,
                       key=f"rnd_{meta['demo_id']}_{player}_{e['kill_id'] if e is not None else ''}_{nonce}")
    ctx = round_ctx(str(d["work"]), rnd)
    tmin, tmax = int(ctx["tick"].min()), int(ctx["tick"].max())
    default_tick = e["kill_tick"] if (e is not None and e["round"] == rnd) else tmin
    # çok katlı harita (yalnız gösterim). Olay seçilince saldırganın katı otomatik; elle seçim başka olaya kadar kalır.
    lvl_on = V.level_enabled(info, meta["map"])
    sel_level, att_lvl, vic_lvl = "default", None, None
    if lvl_on:
        key_lvl, key_lvl_for = f"level_{meta['demo_id']}", f"level_for_{meta['demo_id']}"
        if e is not None:
            att_lvl, vic_lvl = V.event_levels(round_ctx(str(d["work"]), e["round"]), info, e)
            if st.session_state.get(key_lvl_for) != e["kill_id"]:
                st.session_state[key_lvl] = V.LEVEL_LABEL[att_lvl]
                st.session_state[key_lvl_for] = e["kill_id"]
        st.session_state.setdefault(key_lvl, V.LEVEL_LABEL["default"])
        _lab = st.segmented_control("Kat", list(V.LEVEL_LABEL.values()), key=key_lvl, label_visibility="collapsed")
        sel_level = V.LEVEL_FROM_LABEL.get(_lab or "", att_lvl or "default")
        notes_lvl = []
        if att_lvl and vic_lvl and att_lvl != vic_lvl:
            notes_lvl.append(f"Seçili olayda saldırgan {V.LEVEL_LABEL[att_lvl].lower()}ta, kurban "
                             f"{V.LEVEL_LABEL[vic_lvl].lower()}ta: farklı katlar.")
        if att_lvl and att_lvl != sel_level:
            notes_lvl.append(f"Seçili olayın saldırganı {V.LEVEL_LABEL[att_lvl].lower()}ta; şu an "
                             f"{V.LEVEL_LABEL[sel_level].lower()} gösteriliyor.")
        notes_lvl.append("Soluk / içi boş işaret = diğer kattaki oyuncu (konumu bu kat görselinde değildir).")
        st.markdown('<div class="fltsum">' + "<br>".join(esc(n) for n in notes_lvl) + "</div>", unsafe_allow_html=True)
    other_lbl = V.LEVEL_LABEL["lower" if sel_level == "default" else "default"].lower()
    if mode == MODE_PLAY:
        # round oynatma — yalnız map_context/utility okur; skor, olay, oyuncu ve filtre durumuna dokunmaz
        tick = default_tick
        pfig, tech = playback_fig(str(d["work"]), rnd, None, PB.PLAYBACK_TRAIL_S, RADAR_H,
                                  sel_level if lvl_on else None)
        c2.markdown(f'<div class="note" style="padding-top:30px">Round {tech["round"]} · {tech["seconds"]} sn · '
                    f'{tech["frames"]} kare · {tech["hz"]} Hz · iz {tech["trail_s"]:g} sn · oynatma tarayıcıda; '
                    'skorlar ve olay seçimi değişmez'
                    + (' · <b>Uzun round için güvenli oynatma modu</b>' if tech.get("safe_mode") else "")
                    + '</div>', unsafe_allow_html=True)
        st.plotly_chart(pfig, width="stretch", config={"displaylogo": False},
                        key=f"pb_{meta['demo_id']}_{rnd}_{sel_level}")
        st.caption("Yakın oyuncu etiketleri birleşir (ör. 'Player_2 +1'); ad, noktanın üzerine gelince görünür.")
        if info.get("image_missing"):
            st.caption(RADAR_MISSING)
    else:
        tick = c2.slider("Tick (round içinde)", tmin, tmax, int(min(max(default_tick, tmin), tmax)), step=8,
                         key=f"tick_{meta['demo_id']}_{player}_{e['kill_id'] if e is not None else ''}_{rnd}_{nonce}")

        def px(x, y):
            return R.world_to_pixel(info, x, y)

        fig = go.Figure()
        bg_img = (next(lv["image"] for lv in info["levels"] if lv["name"] == sel_level) if lvl_on
                  else info["levels"][0]["image"])
        if bg_img:
            fig.add_layout_image(dict(source=radar_image(bg_img), x=0, y=0, sizex=info["size"],
                                      sizey=info["size"], xref="x", yref="y", sizing="stretch", layer="below", opacity=0.92))
        else:                                       # radar görseli yok: boş zemin + ince çerçeve
            fig.add_shape(type="rect", x0=0, y0=0, x1=info["size"], y1=info["size"], layer="below",
                          line=dict(color="#2f3a48", width=1), fillcolor="#11161d")
        util = V.active_utility(d["utility"], tick)
        for u in util.iter_rows(named=True):
            cx, cy = px(u["x"], u["y"])
            r = (V.SMOKE_RADIUS if u["kind"] == "smoke" else V.INFERNO_RADIUS) / info["scale"]
            u_other = lvl_on and (("lower" if (u["z"] or 0) < V.level_boundary(info) else "default") != sel_level)
            fig.add_shape(type="circle", x0=cx - r, x1=cx + r, y0=cy - r, y1=cy + r,
                          line=dict(width=1, dash="dot" if u_other else "solid",
                                    color="rgba(255,255,255,0.35)" if u["kind"] == "smoke" else "rgba(235,104,52,0.5)"),
                          fillcolor="rgba(0,0,0,0)" if u_other else (SMOKE if u["kind"] == "smoke" else FIRE))
        for kind, col, name in (("smoke", SMOKE, "duman"), ("molotof", FIRE, "molotof")):   # lejant girdisi
            if util.height and (util["kind"] == kind).any():
                fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name=name,
                                         marker=dict(size=12, color=col, line=dict(color="white", width=1))))
        sel = e is not None and e["round"] == rnd
        focus = {player, e["victim"]} if sel else {player}
        for p, g in V.live_trails(ctx, tick).items():          # yollar (yalnız 'live')
            x, y = px(g["X"].to_numpy(), g["Y"].to_numpy())
            if p == player:
                col, w, op = ATTACKER, 2.2, 0.9
            elif sel and p == e["victim"]:
                col, w, op = VICTIM, 2.2, 0.9
            else:
                col, w, op = (OTHER_T if g["team_num"][0] == 2 else OTHER_CT), 1, 0.3
            if lvl_on:                                   # izin diğer kattaki bölümü soluk ve noktalı
                (sx, sy), (ox_, oy_) = V.trail_segments(g, info, sel_level)
                sx, sy = px(pl.Series(sx, dtype=pl.Float64).to_numpy(), pl.Series(sy, dtype=pl.Float64).to_numpy())
                ox_, oy_ = px(pl.Series(ox_, dtype=pl.Float64).to_numpy(), pl.Series(oy_, dtype=pl.Float64).to_numpy())
                fig.add_trace(go.Scatter(x=sx, y=sy, mode="lines", line=dict(color=col, width=w), opacity=op,
                                         hoverinfo="skip", showlegend=False))
                fig.add_trace(go.Scatter(x=ox_, y=oy_, mode="lines", line=dict(color=col, width=1, dash="dot"),
                                         opacity=0.18, hoverinfo="skip", showlegend=False))
                continue
            fig.add_trace(go.Scatter(x=x, y=y, mode="lines", line=dict(color=col, width=w), opacity=op,
                                     hoverinfo="skip", showlegend=False))
        now = V.positions_at(ctx, tick).filter(pl.col("is_alive"))
        lv_now = ({p_: V.player_level(ctx, info, p_, tick) for p_ in now["player"].to_list()} if lvl_on
                  else {p_: "default" for p_ in now["player"].to_list()})
        me = now.filter(pl.col("player") == player)
        me_other = lvl_on and me.height and lv_now.get(player, "default") != sel_level
        # etiketi ayrıca yazılan seçili oyuncu/kurban konumları (yakın etiketler sadeleşir, noktalar kalır)
        reserved = ([px(e["attacker_X"], e["attacker_Y"]), px(e["victim_X"], e["victim_Y"])] if sel
                    else [tuple(v[0] for v in px(me["X"].to_numpy(), me["Y"].to_numpy()))] if me.height else [])
        others_all = now.filter(~pl.col("player").is_in(list(focus)))
        off_level = [p_ for p_ in others_all["player"].to_list() if lv_now.get(p_, "default") != sel_level]
        others = others_all.filter(~pl.col("player").is_in(off_level))
        ghosts = others_all.filter(pl.col("player").is_in(off_level))
        ox, oy = px(others["X"].to_numpy(), others["Y"].to_numpy())
        otext, res_hidden = V.declutter_labels(ox, oy, others["player"].to_list(), reserved=reserved)
        others = others.with_columns(px_=pl.Series(ox), py_=pl.Series(oy), label=pl.Series(otext, dtype=pl.Utf8))
        plus = [f" +{k}" if k else "" for k in res_hidden] + ["", ""]
        for team, col, name, sym in ((2, OTHER_T, "T", "circle"), (3, OTHER_CT, "CT", "diamond")):
            g = others.filter(pl.col("team_num") == team)
            fig.add_trace(go.Scatter(x=g["px_"].to_numpy(), y=g["py_"].to_numpy(), mode="markers+text",
                                     text=g["label"].to_list(), hovertext=g["player"].to_list(),
                                     hovertemplate="%{hovertext}<extra></extra>",
                                     textposition="top center", textfont=dict(size=9, color="#c9d1d9"), name=name,
                                     marker=dict(size=8, color=col, symbol=sym, line=dict(color="#0b0f14", width=1))))
        if lvl_on and ghosts.height:                    # diğer kattaki oyuncular: soluk, içi boş, yazısız (hover'da ad)
            for team, col, sym in ((2, OTHER_T, "circle-open"), (3, OTHER_CT, "diamond-open")):
                g = ghosts.filter(pl.col("team_num") == team)
                gx, gy = px(g["X"].to_numpy(), g["Y"].to_numpy())
                fig.add_trace(go.Scatter(x=gx, y=gy, mode="markers", opacity=0.45, showlegend=False,
                                         hovertext=[f"{n} ({other_lbl})" for n in g["player"].to_list()],
                                         hovertemplate="%{hovertext}<extra></extra>",
                                         marker=dict(size=9, color=col, symbol=sym, line=dict(width=1.5))))
        if lvl_on:
            fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name=f"içi boş = {other_lbl}",
                                     marker=dict(size=10, color=GRAY, symbol="circle-open", line=dict(width=1.5))))
        if me.height and not sel:      # seçili oyuncu o anda (olay round'unda değilse)
            x, y = px(me["X"].to_numpy(), me["Y"].to_numpy())
            fig.add_trace(go.Scatter(x=x, y=y, mode="markers+text",
                                     text=[player + plus[0] + (f" ({other_lbl})" if me_other else "")],
                                     textposition="top center", opacity=0.5 if me_other else 1,
                                     textfont=dict(color=ATTACKER, size=11), name=player,
                                     marker=dict(size=13, color=ATTACKER, symbol="circle-open" if me_other else "circle",
                                                 line=dict(color="white" if not me_other else ATTACKER, width=1.5))))
            ex, ey = px(*V.aim_endpoint(me["X"][0], me["Y"][0], me["yaw"][0]))
            fig.add_annotation(x=ex, y=ey, ax=x[0], ay=y[0], xref="x", yref="y", axref="x", ayref="y", showarrow=True,
                               arrowhead=3, arrowsize=1.1, arrowwidth=2.2, arrowcolor=ATTACKER)
        if sel:
            a_other = lvl_on and att_lvl is not None and att_lvl != sel_level
            v_other = lvl_on and vic_lvl is not None and vic_lvl != sel_level
            for role, col, nm, oth in (("attacker", ATTACKER, "saldırgan son 2 sn", a_other),
                                       ("victim", VICTIM, "kurban son 2 sn", v_other)):
                w = V.kill_window(d["kill_windows"], e["kill_id"], role)
                if w.height:
                    x, y = px(w["X"].to_numpy(), w["Y"].to_numpy())
                    fig.add_trace(go.Scatter(x=x, y=y, mode="lines", opacity=0.3 if oth else 1,
                                             line=dict(color=col, width=2 if oth else 4, dash="dot" if oth else "solid"),
                                             name=nm + (f" ({other_lbl})" if oth else "")))
            ax_, ay_ = px(e["attacker_X"], e["attacker_Y"])
            vx, vy = px(e["victim_X"], e["victim_Y"])
            wb = (e["wallbang"] or 0) > 0
            shot_txt = " · ".join(t for t, on in (("wallbang", wb), ("duman içinden", e["thrusmoke"])) if on)
            fig.add_trace(go.Scatter(x=[ax_, vx], y=[ay_, vy], mode="lines",
                                     line=dict(color="rgba(255,255,255,0.85)", width=2, dash="dash" if wb else "dot"),
                                     name="atış hattı" + (f" ({shot_txt})" if shot_txt else ""), hoverinfo="skip"))
            if shot_txt:
                fig.add_annotation(x=(ax_ + vx) / 2, y=(ay_ + vy) / 2, text=shot_txt.upper(), showarrow=False,
                                   font=dict(color="#ffffff", size=11), bgcolor="rgba(11,15,20,0.85)",
                                   bordercolor="rgba(255,255,255,0.6)", borderwidth=1, borderpad=3)
            ex, ey = px(*V.aim_endpoint(e["attacker_X"], e["attacker_Y"], e["attacker_yaw"]))
            fig.add_annotation(x=ex, y=ey, ax=ax_, ay=ay_, xref="x", yref="y", axref="x", ayref="y", showarrow=True,
                               arrowhead=3, arrowsize=1.2, arrowwidth=2.5, arrowcolor=ATTACKER)
            fig.add_trace(go.Scatter(x=[vx], y=[vy], mode="markers", name="seçili olay", hoverinfo="skip",
                                     marker=dict(size=34, color="rgba(0,0,0,0)", line=dict(color=MARK, width=2.5))))
            fig.add_trace(go.Scatter(x=[vx], y=[vy], mode="markers+text",
                                     text=[e["victim"] + plus[1] + (f" ({other_lbl})" if v_other else "")],
                                     textposition="bottom center", opacity=0.5 if v_other else 1,
                                     textfont=dict(color=VICTIM, size=11), name=f"kurban {e['victim']}",
                                     marker=dict(size=14, color=VICTIM, symbol="x-open" if v_other else "x",
                                                 line=dict(color="white" if not v_other else VICTIM, width=1.5))))
            fig.add_trace(go.Scatter(x=[ax_], y=[ay_], mode="markers+text",
                                     text=[player + plus[0] + (f" ({other_lbl})" if a_other else "")],
                                     textposition="top center", opacity=0.5 if a_other else 1,
                                     textfont=dict(color=ATTACKER, size=11), name=f"saldırgan {player} (ok: bakış)",
                                     marker=dict(size=14, color=ATTACKER, symbol="circle-open" if a_other else "circle",
                                                 line=dict(color="white" if not a_other else ATTACKER, width=1.5))))
            if e["review_point"]:
                fig.add_trace(go.Scatter(x=[vx], y=[vy - 26], mode="markers", name="inceleme noktası",
                                         marker=dict(size=16, color=MARK, symbol="star", line=dict(color="#0b0f14", width=1))))
        fig.update_xaxes(range=[0, info["size"]], visible=False)
        fig.update_yaxes(range=[info["size"], 0], visible=False, scaleanchor="x")
        fig.update_layout(height=RADAR_H, margin=dict(l=0, r=0, t=0, b=50), plot_bgcolor="#0b0f14", paper_bgcolor="#0b0f14",
                          legend=dict(orientation="h", bgcolor="rgba(0,0,0,0)", font=dict(color="#c9d1d9", size=11),
                                      x=0.5, xanchor="center", y=-0.005, yanchor="top"), dragmode="pan")
        st.plotly_chart(fig, width="stretch", config={"displaylogo": False, "scrollZoom": True})
        if info.get("image_missing"):
            st.caption(RADAR_MISSING)
        if meta["map"] in V.SINGLE_LEVEL_NOTE:
            st.caption(V.SINGLE_LEVEL_NOTE[meta["map"]])

# ---------------------------------------------------------------- alt: olay zaman çizelgesi
st.markdown('<div class="sec">Olay zaman çizelgesi · bütün öldürmeler</div>', unsafe_allow_html=True)
if fe.height == 0:
    st.info(f"Seçili filtrelere uyan olay yok (0 / {d['events'].height}). Filtreleri gevşetin veya temizleyin.")
    st.button("Filtreleri temizle", key=f"fltc2_{meta['demo_id']}", on_click=_clear_filters)
else:
    tf = go.Figure()
    for i, r in enumerate(rt):                       # round bantları (dönüşümlü gölge); yalnız canlı round'lar
        if i % 2 == 0:
            tf.add_vrect(x0=r["start_tick"], x1=r["end_tick"], fillcolor="rgba(255,255,255,0.03)", line_width=0, layer="below")
    tl = fe.with_columns(me=(pl.col("attacker") == player))
    for mine, col, name in ((False, "rgba(139,148,158,0.55)", "diğer oyuncular"), (True, ATTACKER, player)):
        g = tl.filter(pl.col("me") == mine)
        rp = g["review_point"].to_list()
        tf.add_trace(go.Scatter(x=g["kill_tick"].to_numpy(), y=g["event_score"].to_numpy(), mode="markers", name=name,
                                customdata=g["kill_id"].to_numpy(), unselected=dict(marker=dict(opacity=1)),
                                marker=dict(color=col, size=[12 if r else 7 for r in rp],
                                            symbol=["star" if r else "circle" for r in rp], line=dict(width=0)),
                                text=[f"{a} → {v} · R{r}" for a, v, r in zip(g["attacker"], g["victim"], g["round"])],
                                hovertemplate="%{text}<br>tick %{x}<br>olay skoru %{y:.3f}<extra></extra>"))
    if e is not None:
        tf.add_trace(go.Scatter(x=[e["kill_tick"]], y=[e["event_score"]], mode="markers", name="seçili olay",
                                customdata=[e["kill_id"]], unselected=dict(marker=dict(opacity=1)),
                                marker=dict(size=20, color="rgba(0,0,0,0)", line=dict(color=MARK, width=2.5))))
    tf.add_vline(x=tick, line=dict(color=MARK, width=1, dash="dot"))
    if rt:   # eksen: round numaraları, en fazla ~16 etiket (çakışma olmasın)
        step = max(1, -(-len(rt) // 16))
        tf.update_xaxes(tickvals=[(r["start_tick"] + r["end_tick"]) / 2 for r in rt[::step]],
                        ticktext=[f"R{r['round']}" for r in rt[::step]], tickangle=0,
                        range=[rt[0]["start_tick"], rt[-1]["end_tick"]])
    tf.update_xaxes(showgrid=False, zeroline=False, tickfont=dict(size=10, color=GRAY), automargin=True)
    tf.update_yaxes(title=dict(text="olay skoru", font=dict(size=11, color=GRAY)), gridcolor="#1c2430",
                    zeroline=False, tickfont=dict(size=10, color=GRAY), range=[0, 1.05], automargin=True)
    tf.update_layout(height=220, margin=dict(l=10, r=10, t=30, b=10), plot_bgcolor="#0b0f14", paper_bgcolor="#0b0f14",
                     legend=dict(orientation="h", x=1, xanchor="right", y=1.18, font=dict(color="#c9d1d9", size=11),
                                 bgcolor="rgba(0,0,0,0)"))
    key_tl = f"tl_{meta['demo_id']}_{nonce}"


    def _on_timeline_select():
        """Tıklanan noktayı bekleyen seçim olarak sakla; grafik anahtarını yenile (seçim temizlenir → aynı noktaya tekrar
        tıklama da çalışır). Seçim bir sonraki çalıştırmanın başında, widget'lardan önce uygulanır."""
        pts = (st.session_state.get(key_tl) or {}).get("selection", {}).get("points", [])
        pick = V.timeline_pick(pts, d["events"])
        if pick:
            st.session_state[key_pick] = pick
            st.session_state[key_nonce] = nonce + 1


    tf.update_layout(clickmode="event+select", dragmode=False)
    st.plotly_chart(tf, width="stretch", config={"displaylogo": False}, key=key_tl, on_select=_on_timeline_select,
                    selection_mode="points")
    st.caption("Noktaya tıklayın: olay açılır. ★ inceleme noktası · sarı halka: seçili olay · noktalı çizgi: tick")

# ---------------------------------------------------------------- alt bölümler: detaylar ve sınırlılıklar
with st.expander("Detaylı veriler"):
    t = V.detail_tables(d, player)
    st.markdown("**Oyuncular (model çıktısı)**")
    st.dataframe(t["players"].to_pandas(), hide_index=True, width="stretch")
    st.markdown(f"**{player} — olaylar (olay skoruna göre)**")
    st.dataframe(t["events"].to_pandas(), hide_index=True, width="stretch", height=260)
    st.markdown("**Analiz özeti**")
    st.markdown(kv([("Harita", meta["map"]), ("Round", meta.get("rounds")), ("Süre (dk)", V.fmt_value(meta.get("minutes"), 1)),
                    ("Modele giren öldürme", meta.get("kills_model", V.UNKNOWN)), ("Kaynak", meta.get("source")),
                    ("demo_id", meta["demo_id"])]), unsafe_allow_html=True)
if d["labels"] is not None:
    with st.expander("Veri seti etiketi (CS2CD) — model çıktısı DEĞİLDİR"):
        st.dataframe(d["labels"].rename({"player": "oyuncu", "dataset_label": "veri seti etiketi"}).to_pandas(),
                     hide_index=True, width="stretch")
with st.expander("Model sınırlılıkları"):
    lim = V.limitations(entry["mode"], d["result"], bold=True)
    st.markdown("\n".join(f"- {x}" for x in lim))
    if dist:
        import distribution_check as DC
        st.markdown(DC.markdown(dist, heading="####"))
