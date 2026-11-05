"""Faz 3.5: maç inceleme raporu — tek, taşınabilir HTML (tarayıcıdan PDF'e yazdırılabilir).

YALNIZ mevcut analiz çıktılarından (reports/analysis/<id>/ veya samples/analysis/<id>/): rapor üretimi sırasında yeni
model çıkarımı, yeni skor veya yeni eşik UYGULANMAZ. Eşikler models/v1'den yalnız okunup gösterilir.
Gizlilik denetimi ZORUNLU: Steam ID deseni, .dem adı, kimlik eşlemesi, yerel dosya yolu veya Player_N dışı oyuncu adı
bulunursa rapor yazılmaz (ReportPrivacyError).
Radar görselleri Pillow ile statik çizilir (Kaleido/Chrome yok); radar dosyası yoksa boş zemin.

    python src/report_export.py <analiz_klasörü> [--compare Player_6 Player_3] [-o çıktı.html]
"""
import argparse
import base64
import io
import json
import re
from datetime import datetime
from pathlib import Path

import polars as pl
from jinja2 import Environment
from PIL import Image, ImageDraw

import distribution_check as DC
import radar as R
import viewer_data as V
from score_demo import DISCLAIMER, EVENT_NOTE

EXPORT_DIR = Path("reports/exports")
MAX_EVENTS = 10
MAX_EVENT_IMAGES = 6
IMG_PX = 420
PROVENANCE = ("Bu rapor mevcut Model v1 analiz çıktılarından oluşturulmuştur; rapor üretimi sırasında yeni model "
              "çıkarımı, yeni skor veya yeni eşik uygulanmaz.")
LABEL_TITLE = "Veri seti etiketi — model çıktısı değildir"
# Raporda Türkçe terim önde (arayüz metni değişmez). Sıra önemli: uzun ifade önce.
TERMS = [
    ("professional/high-skill false-positive risk",
     "Profesyonel / yüksek beceri yanlış alarm riski (professional/high-skill false-positive risk)"),
    ("Pre-FOV açısı (rakip görünmeden ~190 ms önce)", "Görüş öncesi nişan açısı (pre-FOV; rakip görünmeden ~190 ms önce)"),
    ("Pre-FOV açısı", "Görüş öncesi nişan açısı (pre-FOV)"),
    ("pre-FOV açısı", "görüş öncesi nişan açısı (pre-FOV)"),
    ("Wallbang öldürme", "Duvar arkası isabetli öldürme (wallbang)"),
    ("Görünürlük ray-cast ile", "Görünürlük ışın izleme (ray-cast) ile"),
]
# Sınırlılıklar: kritik olanlar ana listede, teknik/ikincil olanlar "Ek sınırlılıklar"da (içerik silinmez)
PRIMARY_LIMITS = (V.PRO_FP_LIMIT, V.LOW_CONF, V.SNIPER_LIMIT, EVENT_NOTE)
SUMMARY_KEYS = ["reaction_ms", "prefov_angle", "inv_angle_med", "not_visible", "wallbang", "thrusmoke"]
SUMMARY_HEAD = [("Ortalama tepki", "düşük = daha kısa"), ("Görüş öncesi nişan açısı (pre-FOV)", "düşük = hedefe yakın"),
                ("Görünmezken nişan açısı", "düşük = hedefe yakın"), ("Görünür olmayan öldürme", "sayı / bilinen (oran)"),
                ("Duvar arkası (wallbang)", "sayı / öldürme (oran)"), ("Duman içinden", "sayı / öldürme (oran)")]


class ReportPrivacyError(Exception):
    """Rapor gizlilik denetiminden geçmedi; dosya yazılmadı."""


# ---------------------------------------------------------------- gizlilik
_STEAMID = re.compile(r"7656119\d{10}")
_PATHS = re.compile(r"(?i)([a-z]:[\\/])|(/users/)|(/home/)|(\\users\\)")
_PLAYER = re.compile(r"^(|Player_\d+)$")


def privacy_problems_html(html: str) -> list[str]:
    """HTML metninde (gömülü görseller çıkarılarak) yasak içerik."""
    text = re.sub(r"data:image/[a-z]+;base64,[A-Za-z0-9+/=]+", "", html)
    probs = []
    if _STEAMID.search(text):
        probs.append("Steam ID deseni")
    if re.search(r"\.dem\b", text):
        probs.append(".dem dosya adı")
    if ".ids.json" in text or "steamid" in text.lower():
        probs.append("kimlik eşlemesi / steamid")
    if _PATHS.search(text):
        probs.append("yerel dosya yolu")
    return probs


def privacy_problems_data(d: dict) -> list[str]:
    """Rapora giren oyuncu adları yalnız anonim Player_N (veya boş = dünya ölümü)."""
    names = set(d["players"]["player"].to_list())
    for c in ("attacker", "victim"):
        names |= set(d["events"][c].drop_nulls().to_list())
    bad = sorted(n for n in names if not _PLAYER.match(str(n)))
    return [f"anonim olmayan oyuncu adı: {len(bad)} adet"] if bad else []


# ---------------------------------------------------------------- veri
def _entry_for(work: Path) -> dict:
    work = Path(work)
    for e in V.list_analyses(work.parent):
        if Path(e["work"]).resolve() == work.resolve():
            return e
    raise FileNotFoundError(f"açılabilir analiz değil (skor yok): {work.name}")


def important_events(d: dict) -> pl.DataFrame:
    """Yalnız MEVCUT bayraklarla seçim: önce inceleme noktaları (olay skoru ↓, kill_id), en çok 10; eksikse yüksek/orta
    öncelikli oyuncuların mevcut en yüksek skorlu olayları (top_event_ids, öncelik sırasıyla)."""
    ev = d["events"]
    rp = ev.filter(pl.col("review_point").fill_null(False)).sort(["event_score", "kill_id"], descending=[True, False])
    ids = rp["kill_id"].to_list()[:MAX_EVENTS]
    if len(ids) < MAX_EVENTS:
        P = d["players"]
        for p in V.player_order(P):
            r = P.filter(pl.col("player") == p).row(0, named=True)
            if r["review_priority"] not in ("yüksek inceleme önceliği", "orta inceleme önceliği"):
                continue
            for k in r.get("top_event_ids") or []:
                if k not in ids and len(ids) < MAX_EVENTS:
                    ids.append(int(k))
    order = {k: i for i, k in enumerate(ids)}
    return (ev.filter(pl.col("kill_id").is_in(ids)).with_columns(_o=pl.col("kill_id").replace_strict(order))
              .sort("_o").drop("_o"))


def event_image(work: Path, map_name: str, e: dict) -> tuple[str, bool]:
    """Olay anının statik radar görseli (Pillow): saldırgan (kırmızı), kurban (mavi), atış hattı, bakış oku.
    Dönüş: (data URI, radar görseli var mı)."""
    info = V.radar_info(work, map_name)
    size = info["size"]
    img_name = info["levels"][0]["image"]
    v_other = False
    if V.level_enabled(info, map_name):            # saldırganın katının görseli; kurban diğer kattaysa içi boş
        a_lvl, v_lvl = V.event_levels(V.context_round(work, e["round"]), info, e)
        img_name = next(lv["image"] for lv in info["levels"] if lv["name"] == a_lvl)
        v_other = v_lvl != a_lvl
    if img_name:
        base = Image.open(R.RADAR_DIR / img_name).convert("RGB").resize((size, size))
    else:
        base = Image.new("RGB", (size, size), (17, 22, 29))
    dr = ImageDraw.Draw(base)
    ax, ay = R.world_to_pixel(info, e["attacker_X"], e["attacker_Y"])
    vx, vy = R.world_to_pixel(info, e["victim_X"], e["victim_Y"])
    ex, ey = R.world_to_pixel(info, *V.aim_endpoint(e["attacker_X"], e["attacker_Y"], e["attacker_yaw"]))
    wb = (e.get("wallbang") or 0) > 0
    if wb:                                              # wallbang: kesikli atış hattı
        n = 14
        for i in range(0, n, 2):
            t0, t1 = i / n, (i + 1) / n
            dr.line([(ax + (vx - ax) * t0, ay + (vy - ay) * t0), (ax + (vx - ax) * t1, ay + (vy - ay) * t1)],
                    fill=(255, 255, 255), width=3)
    else:
        dr.line([(ax, ay), (vx, vy)], fill=(255, 255, 255), width=2)
    dr.line([(ax, ay), (ex, ey)], fill=(255, 77, 79), width=4)                       # bakış oku
    r = 9
    dr.ellipse([ax - r, ay - r, ax + r, ay + r], fill=(255, 77, 79), outline=(255, 255, 255), width=2)
    if v_other:
        dr.ellipse([vx - r, vy - r, vx + r, vy + r], outline=(59, 139, 255), width=3)
    else:
        dr.line([(vx - r, vy - r), (vx + r, vy + r)], fill=(59, 139, 255), width=5)
        dr.line([(vx - r, vy + r), (vx + r, vy - r)], fill=(59, 139, 255), width=5)
    pad = 170                                                                        # olay çevresine kırp
    x0, x1 = max(0, min(ax, vx, ex) - pad), min(size, max(ax, vx, ex) + pad)
    y0, y1 = max(0, min(ay, vy, ey) - pad), min(size, max(ay, vy, ey) + pad)
    side = max(x1 - x0, y1 - y0, 340)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    box = [int(max(0, min(size - side, cx - side / 2))), int(max(0, min(size - side, cy - side / 2)))]
    crop = base.crop((box[0], box[1], box[0] + int(side), box[1] + int(side))).resize((IMG_PX, IMG_PX))
    buf = io.BytesIO()
    crop.save(buf, format="JPEG", quality=82, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode(), bool(img_name)


def tr_terms(text):
    """Raporda Türkçe terimi öne alır; zaten Türkçe-önde olan metni ikinci kez değiştirmez."""
    if not isinstance(text, str):
        return text
    import re as _re
    for en, trk in TERMS:
        if trk in text:
            continue
        text = _re.sub(_re.escape(en), trk, text, flags=_re.IGNORECASE) if en.islower() else text.replace(en, trk)
    return text


def split_limitations(lims: list[str]) -> tuple[list[str], list[str]]:
    """Ana (kritik) ve ek sınırlılıklar; hiçbir madde atılmaz, sıra korunur."""
    main, extra = [], []
    for x in lims:
        is_main = (x.startswith(DISCLAIMER) or x in PRIMARY_LIMITS or "(OOF)" in x)
        (main if is_main else extra).append(x)
    return main, extra


def _fmt(v, nd=0, unit=""):
    return V.fmt_value(v, nd, unit)


def build_context(work: Path, compare: tuple[str, str] | None = None, generated_at: str | None = None) -> dict:
    work = Path(work)
    entry = _entry_for(work)
    d = V.load(entry)
    meta, P, ev = d["meta"], d["players"], d["events"]
    probs = privacy_problems_data(d)
    if probs:
        raise ReportPrivacyError("; ".join(probs))
    order = V.player_order(P)
    prio_counts = {k: P.filter(pl.col("review_priority") == k).height for k in V.PRIORITY_ORDER}
    thr = json.loads(Path("models/v1/thresholds.json").read_text(encoding="utf-8"))
    ref = json.loads(Path("data/reference/v1/reference.json").read_text(encoding="utf-8"))
    ident = d["result"].get("model_identity") or {}
    players = []
    for p in order:
        r = P.filter(pl.col("player") == p).row(0, named=True)
        summ = {x["key"]: x for x in V.compare_players(d, p, p)}
        players.append(dict(
            name=p, style=r.get("style") or V.UNKNOWN, kills=_fmt(r.get("analyzed_kills")),
            score=_fmt(r.get("player_raw_score"), 3), pct=V.fmt_pct(r.get("style_percentile")),
            priority=r.get("review_priority") or V.UNKNOWN, points=int(r.get("review_points") or 0),
            low_conf=bool(r.get("low_confidence_style")), explanation=r.get("player_explanation") or "",
            summary=[dict(text=summ[k]["a_text"], pct=summ[k].get("a_pct")) for k in SUMMARY_KEYS]))
    imp = important_events(d)
    events = []
    for i, e in enumerate(imp.iter_rows(named=True)):
        img, has_radar = event_image(work, meta["map"], e) if i < MAX_EVENT_IMAGES else (None, True)
        events.append(dict(
            kill_id=e["kill_id"], round=e["round"], tick=e["kill_tick"], attacker=e["attacker"], victim=e["victim"],
            weapon=e["weapon"], score=_fmt(e["event_score"], 3), pct=V.fmt_pct(e["event_style_percentile"]),
            review_point=bool(e["review_point"]), wallbang=V.yes_no((e["wallbang"] or 0) > 0 if e["wallbang"] is not None
                                                                    else None),
            smoke=V.yes_no(e["thrusmoke"]), visible=V.visibility_text(e["victim_visible_at_kill"]),
            reaction=_fmt(e["reaction_ms"], 0, " ms"), prefov=_fmt(e["prefov_angle"], 1, "°"),
            explanation=e.get("explanation") or "", image=img, has_radar=has_radar))
    cmp = None
    if compare:
        a, b = compare
        if a not in order or b not in order:
            raise ValueError(f"karşılaştırma oyuncusu analizde yok: {a}, {b}")
        rows = V.compare_players(d, a, b)
        sty = {r["key"]: r for r in rows}["style"]
        cmp = dict(a=a, b=b, rows=rows, note=V.COMPARE_NOTE,
                   style_warn=bool(sty["a"] and sty["b"] and sty["a"] != sty["b"]), style_a=sty["a"], style_b=sty["b"])
    dist = V.distribution(d["work"])
    is_cs2cd = entry["mode"] == "cs2cd_oof"
    return dict(
        title=f"İnceleme raporu · {meta['map']} · {meta['demo_id']}",
        generated_at=generated_at or datetime.now().strftime("%Y-%m-%d %H:%M"),
        disclaimer=DISCLAIMER, provenance=PROVENANCE, event_note=EVENT_NOTE,
        summary=[("demo_id", meta["demo_id"]),
                 ("Kaynak", "CS2CD eğitim havuzu (OOF skorları)" if is_cs2cd else (meta.get("source") or V.UNKNOWN)),
                 ("Harita", meta["map"]), ("Round", meta.get("rounds")),
                 ("Süre", _fmt(meta.get("minutes"), 1, " dk")), ("Modele giren öldürme", meta.get("kills_model")),
                 ("Oyuncu", P.height)] + ([("CS2CD maçı", entry["match"])] if is_cs2cd else []),
        rounds=V.live_rounds(meta),
        model=[("Model", "Model B v1 (dondurulmuş, models/v1/)"),
               ("model_B.joblib sha256", (ident.get("models/v1/model_B.joblib") or "")[:16] + "…"),
               ("Skor sürümü", d["result"].get("scoring_version") or d["result"].get("oof_protocol") or V.UNKNOWN),
               ("Varsayılan karar", "G1: stil içi yüzdelik + inceleme önceliği; G0 yalnız karşılaştırma"),
               ("G1 oyuncu eşikleri (stil içi temizlerin %95'lik dilimi)",
                ", ".join(f"{k} > {v:.3f}" for k, v in thr["G1"].items())),
               ("Olay eşiği (inceleme noktası)", ", ".join(f"{k} > {v:.3f}" for k, v in ref["event_threshold"].items())),
               ("Öncelik tanımı", "yüksek = skor > G1 eşiği; orta = stil içi temizlerin %90'lık dilimi üstü; düşük = "
                                  "altı; < 3 öldürme = yetersiz veri")],
        is_cs2cd=is_cs2cd, mode_note=d["result"].get("mode_note"),
        dist=dist, dist_reasons=DC.reasons(dist) if dist and dist["level"] > 0 else "",
        prio_counts=prio_counts, players=players, summary_head=SUMMARY_HEAD, events=events,
        n_review_points=int(ev["review_point"].fill_null(False).sum()), max_events=MAX_EVENTS,
        max_images=MAX_EVENT_IMAGES, cmp=cmp,
        limitations=split_limitations(V.limitations(entry["mode"], d["result"])),
        labels=(d["labels"].to_dicts() if d["labels"] is not None else None), label_title=LABEL_TITLE,
        radar_missing=not any(e["has_radar"] for e in events[:MAX_EVENT_IMAGES]) if events else False,
        radar_missing_text=V.RADAR_MISSING)


# ---------------------------------------------------------------- HTML
TEMPLATE = r"""<!doctype html>
<html lang="tr"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{{ title }}</title>
<style>
:root { --ink:#1b2129; --muted:#5b6673; --line:#d7dde4; --soft:#f3f5f8; --red:#c62828; --orange:#c77700;
        --green:#2e7d32; --gray:#6b7280; --blue:#1f5fbf; --a:#7e57c2; --b:#00897b; }
* { box-sizing:border-box; }
body { font-family: "Segoe UI", system-ui, -apple-system, sans-serif; color:var(--ink); background:#fff; margin:0;
       font-size:13px; line-height:1.45; }
main { max-width: 980px; margin: 0 auto; padding: 28px 24px 40px; }
h1 { font-size:22px; margin:0 0 4px; } h2 { font-size:16px; margin:26px 0 8px; border-bottom:2px solid var(--line);
     padding-bottom:4px; } h3 { font-size:14px; margin:14px 0 6px; }
.muted { color:var(--muted); } .small { font-size:11.5px; }
.warn { border:2px solid var(--ink); background:#fffbe6; padding:10px 14px; border-radius:6px; margin:12px 0; }
.warn b { font-size:14px; }
.prov { border-left:4px solid var(--blue); background:var(--soft); padding:8px 12px; margin:8px 0; }
.dist { border-left:4px solid var(--gray); background:var(--soft); padding:8px 12px; margin:8px 0; }
table { width:100%; border-collapse:collapse; margin:6px 0 10px; }
th, td { border-bottom:1px solid var(--line); padding:5px 7px; text-align:left; vertical-align:top; }
th { background:var(--soft); font-weight:600; font-size:12px; }
table.kv td:first-child { width:38%; color:var(--muted); }
.badge { display:inline-block; padding:1px 8px; border-radius:10px; color:#fff; font-size:11px; font-weight:700;
         white-space:nowrap; }
.b-yuksek { background:var(--red); } .b-orta { background:var(--orange); } .b-dusuk { background:var(--green); }
.b-yetersiz { background:var(--gray); }
.counts { display:flex; gap:10px; flex-wrap:wrap; } .counts div { border:1px solid var(--line); border-radius:6px;
          padding:6px 12px; min-width:120px; } .counts b { font-size:18px; display:block; }
.ev { display:grid; grid-template-columns: 220px 1fr; gap:12px; border:1px solid var(--line); border-radius:6px;
      padding:10px; margin:8px 0; }
.ev img { width:220px; height:220px; border-radius:4px; border:1px solid var(--line); }
.labels { border:2px dashed var(--gray); background:#f7f7f7; padding:10px 14px; border-radius:6px; margin-top:22px; }
.labels h2 { border-bottom:none; margin-top:0; color:var(--gray); }
.pa { color:var(--a); font-weight:700; } .pb { color:var(--b); font-weight:700; }
footer { margin-top:28px; border-top:1px solid var(--line); padding-top:8px; }
.keep { break-inside: avoid; page-break-inside: avoid; }          /* başlık + ilk içerik birlikte */
.keep > table:last-child, table.cont { margin-bottom:0; } table.cont { margin-top:0; }
table.compact { table-layout: fixed; font-size:11px; }
table.compact th { font-size:10.5px; padding:4px 5px; line-height:1.25; }
table.compact td { padding:4px 5px; }
table.compact .nw { white-space: nowrap; }
.thnote { display:block; font-weight:400; color:var(--muted); font-size:9.5px; }
.imgnote { font-size:10px; color:#8a939e; font-style:italic; align-self:end; padding-bottom:4px; }
ul.lim li { margin-bottom:4px; } ul.lim2 { font-size:12px; color:var(--muted); margin-bottom:0; }
ul.lim2.cont { margin-top:0; }
tr { break-inside: avoid; page-break-inside: avoid; }
@page { size: A4; margin: 14mm 12mm; }
@media print { main { padding:0; max-width:none; } h2, h3 { break-after: avoid; page-break-after: avoid; }
  .ev, .warn, .counts, .labels, .dist, .prov, table.kv { break-inside: avoid; }
  a { color:inherit; text-decoration:none; } }
</style></head>
<body><main>
<h1>{{ title }}</h1>
<div class="muted small">Oluşturma: {{ generated_at }} · Oyuncular anonimdir (Player_N).</div>
<div class="warn"><b>{{ disclaimer }}</b> Bu rapor kimseyi "hileci" olarak nitelendirmez; yalnızca hangi oyuncu ve
  olaylara önce bakılabileceğini (inceleme önceliği) gösterir.</div>
<div class="prov">{{ provenance }}</div>

<div class="keep"><h2>1. Analiz özeti</h2>
<table class="kv">{% for k, v in summary %}<tr><td>{{ k }}</td><td>{{ v }}</td></tr>{% endfor %}</table></div>
{% if mode_note %}<p class="small muted">{{ mode_note }}: bu maç modelin eğitim havuzunda; skorlar maçı görmemiş kat
  modellerinden (out-of-fold).</p>{% endif %}
<div class="keep"><h3>Round bilgileri</h3>
<p class="small">{{ rounds|length }} canlı round.
{% for r in rounds %}R{{ r.round }}{% if not loop.last %}, {% endif %}{% endfor %}</p></div>

<div class="keep"><h2>2. Model v1 bilgisi</h2>
<table class="kv">{% for k, v in model %}<tr><td>{{ k }}</td><td>{{ v }}</td></tr>{% endfor %}</table></div>

<div class="keep"><h2>3. Demo düzeyi dağılım bilgisi</h2>
{% if dist and dist.level > 0 %}
<div class="dist"><b>{{ dist.title }}</b><br>{{ dist.text }}<br><span class="small">Neden: {{ dist_reasons }}.</span>
  <br><span class="small muted">{{ dist.note|tr }}</span></div>
{% elif dist %}<p>{{ dist.text }} <span class="small muted">{{ dist.note|tr }}</span></p>
{% else %}<p class="muted">Dağılım bilgisi hesaplanamadı.</p>{% endif %}</div>

<div class="keep"><h2>4. İnceleme öncelikleri</h2>
<div class="counts">{% for k, n in prio_counts.items() %}<div><b>{{ n }}</b>{{ k }}</div>{% endfor %}</div></div>

<div class="keep"><h2>5. Oyuncular (model çıktısı)</h2>
<table><tr><th>oyuncu</th><th>stil</th><th>öldürme</th><th>model skoru</th><th>stil içi yüzdelik</th>
<th>inceleme önceliği</th><th>inc. noktası</th></tr>
{% for p in players %}<tr><td><b>{{ p.name }}</b></td><td>{{ p.style }}{% if p.low_conf %} (düşük güven){% endif %}</td>
<td>{{ p.kills }}</td><td>{{ p.score }}</td><td>{{ p.pct }}</td>
<td><span class="badge {{ 'b-yuksek' if p.priority.startswith('yüksek') else 'b-orta' if p.priority.startswith('orta')
  else 'b-dusuk' if p.priority.startswith('düşük') else 'b-yetersiz' }}">{{ p.priority }}</span></td>
<td>{{ p.points }}</td></tr>{% endfor %}</table></div>
{% for p in players %}{% if loop.first %}<div class="keep"><h3>Kısa açıklamalar (mevcut model açıklaması)</h3>
<table>{% elif loop.index == 2 %}<table class="cont">{% endif %}<tr><td style="width:90px"><b>{{ p.name }}</b></td>
<td class="small">{{ p.explanation|tr }}</td></tr>{% if loop.first %}</table></div>{% elif loop.last %}</table>{% endif %}
{% endfor %}

{% set cols %}<colgroup><col style="width:11%"><col style="width:12%"><col style="width:15%"><col style="width:14%">
<col style="width:16%"><col style="width:16%"><col style="width:16%"></colgroup>{% endset %}
{% for p in players %}{% if loop.first %}<div class="keep"><h2>6. Davranış özetleri</h2>
<p class="small muted">Modele giren bütün öldürmeler üzerinden. Altta küçük: aynı stildeki temiz eğitim oyuncularına göre
  yüzdelik (0 = en düşük, 100 = en yüksek). Yüzdelik iyi/kötü ya da hile anlamı taşımaz. Görünür olmamak otomatik
  olarak wallhack anlamına gelmez.</p>
<table class="compact">{{ cols }}<tr><th>oyuncu</th>{% for h, n in summary_head %}<th>{{ h }}<span class="thnote">{{ n }}</span>
</th>{% endfor %}</tr>{% elif loop.index == 2 %}<table class="compact cont">{{ cols }}{% endif %}
<tr><td><b>{{ p.name }}</b></td>{% for s in p.summary %}<td><span class="nw">{{ s.text }}</span>{% if s.pct is not none %}
<div class="small muted">%{{ '%.0f'|format(s.pct) }} yüzdelik</div>{% endif %}</td>{% endfor %}</tr>
{% if loop.first %}</table></div>{% elif loop.last %}</table>{% endif %}{% endfor %}

<div class="keep"><h2>7. Seçilmiş önemli olaylar</h2>
<p class="small muted">Seçim yalnız mevcut bayraklarla: önce inceleme noktaları (bu demoda {{ n_review_points }}),
  olay skoruna göre en çok {{ max_events }}; eksikse yüksek/orta öncelikli oyuncuların en yüksek skorlu olayları.
  İlk {{ max_images }} olay için statik radar görseli (kırmızı = saldırgan ve bakış yönü, mavi × = kurban, beyaz
  çizgi = atış hattı; kesikli = wallbang). {{ event_note }}</p>
{% if radar_missing %}<p class="small">{{ radar_missing_text }}</p>{% endif %}
{% if not events %}</div>{% endif %}
{% for e in events %}<div class="ev">
{% if e.image %}<img src="{{ e.image }}" alt="olay #{{ e.kill_id }} radar">{% else %}<div class="imgnote">
  Radar görseli yalnız ilk {{ max_images }} olay için eklenir.</div>{% endif %}
<div><b>#{{ e.kill_id }} · R{{ e.round }} · tick {{ e.tick }}</b> — {{ e.attacker }} → {{ e.victim }} ({{ e.weapon }})
<table class="kv small">
<tr><td>Olay skoru / stil içi olay yüzdeliği</td><td>{{ e.score }} / {{ e.pct }}</td></tr>
<tr><td>İnceleme noktası</td><td>{{ 'evet' if e.review_point else 'hayır' }}</td></tr>
<tr><td>Duvar arkası (wallbang) / duman içinden</td><td>{{ e.wallbang }} / {{ e.smoke }}</td></tr>
<tr><td>Görünürlük (ışın izleme / ray-cast)</td><td>{{ e.visible }}</td></tr>
<tr><td>Tepki süresi / görüş öncesi nişan açısı (pre-FOV)</td><td>{{ e.reaction }} / {{ e.prefov }}</td></tr>
</table><div class="small">{{ e.explanation|tr }}</div></div></div>{% if loop.first %}</div>{% endif %}{% else %}
<p class="muted">Seçilecek olay yok.</p>{% endfor %}

{% if cmp %}
<div class="keep"><h2>8. Oyuncu karşılaştırması: <span class="pa">A · {{ cmp.a }}</span> –
  <span class="pb">B · {{ cmp.b }}</span></h2>
<p class="small muted">{{ cmp.note }}</p></div>
{% if cmp.style_warn %}<p class="small">Oyuncuların stili farklı (A: {{ cmp.style_a }}, B: {{ cmp.style_b }}).
  Yüzdelikler her oyuncunun kendi stilindeki temiz referansa göredir; iki oyuncu birbirine göre yorumlanmaz.</p>{% endif %}
<table><tr><th>metrik</th><th class="pa">A</th><th class="pb">B</th><th>yön</th></tr>
{% for r in cmp.rows %}<tr><td>{{ r.label|tr }}{% if r.note %}<div class="small muted">{{ r.note }}</div>{% endif %}</td>
<td>{{ r.a_text }}{% if r.a_pct is not none and r.key != 'style_percentile' %}<div class="small muted">
  %{{ '%.0f'|format(r.a_pct) }} yüzdelik</div>{% endif %}</td>
<td>{{ r.b_text }}{% if r.b_pct is not none and r.key != 'style_percentile' %}<div class="small muted">
  %{{ '%.0f'|format(r.b_pct) }} yüzdelik</div>{% endif %}</td><td class="small">{{ r.direction }}</td></tr>{% endfor %}
</table>{% endif %}

<div class="keep"><h2>{{ '9' if cmp else '8' }}. Model sınırlılıkları</h2>
<ul class="lim">{% for x in limitations[0] %}<li>{{ x|tr }}</li>{% endfor %}</ul></div>
{% for x in limitations[1] %}{% if loop.first %}<div class="keep"><h3>Ek sınırlılıklar</h3>
<ul class="lim2"><li>{{ x|tr }}</li></ul></div><ul class="lim2 cont">{% else %}<li>{{ x|tr }}</li>{% endif %}
{% if loop.last %}</ul>{% endif %}{% endfor %}

{% if labels %}
<div class="labels"><h2>{{ label_title }}</h2>
<p class="small">CS2CD veri setinin oyuncu etiketleri (hileli maçlarda etiket gürültülüdür). Model bu etiketleri
  görmez; yukarıdaki skor ve öncelikler bu tablodan bağımsızdır.</p>
<table>{% for r in labels %}<tr><td style="width:90px">{{ r.player }}</td><td>{{ r.dataset_label }}</td></tr>
{% endfor %}</table></div>{% endif %}

<footer class="small muted">{{ disclaimer }} · {{ provenance }}
{% if is_cs2cd %}<br>Veri: CS2CD — Counter-Strike 2 Cheat Detection (Hugging Face
  CS2CD/CS2CD.Counter-Strike_2_Cheat_Detection), CC-BY-4.0.{% endif %}
<br>Ham demo dosyası, Steam ID, gerçek oyuncu adı ve kimlik eşlemesi bu rapora girmez.</footer>
</main></body></html>
"""
_ENV = Environment(autoescape=True, trim_blocks=True, lstrip_blocks=True)
_ENV.filters["tr"] = tr_terms


def build_report(work: Path, compare: tuple[str, str] | None = None, generated_at: str | None = None) -> str:
    """HTML metni. Gizlilik denetimi geçmezse ReportPrivacyError (metin döndürülmez)."""
    html = _ENV.from_string(TEMPLATE).render(**build_context(work, compare, generated_at))
    probs = privacy_problems_html(html)
    if probs:
        raise ReportPrivacyError("; ".join(probs))
    return html


def main(argv=None):
    ap = argparse.ArgumentParser(description="HTML inceleme raporu (yalnız mevcut çıktılardan)")
    ap.add_argument("work")
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"))
    ap.add_argument("-o", "--out")
    a = ap.parse_args(argv)
    work = Path(a.work)
    html = build_report(work, tuple(a.compare) if a.compare else None)
    out = Path(a.out) if a.out else EXPORT_DIR / f"inceleme_raporu_{work.name}.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(f"{out} ({out.stat().st_size / 1e3:.0f} KB)")


if __name__ == "__main__":
    main()
