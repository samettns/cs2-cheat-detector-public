"""Faz 3 arayüzünün veri katmanı (Streamlit'ten bağımsız, test edilebilir).

Yalnız önceden analiz edilmiş klasörleri okur (reports/analysis/<demo_id>/); demo YENİDEN AYRIŞTIRILMAZ.
  - Güncel demolar: Faz 2 çıktıları (player_scores / event_scores / map_events) — veri seti etiketi yok.
  - CS2CD eğitim maçları: oof_* çıktıları + dataset_labels (model çıktısından ayrı alan).
Konumlar: map_context.parquet (8 Hz), kill_windows.parquet (öldürme öncesi 2 sn), utility.parquet (duman/molotof).
Oyuncu yolları yalnız 'live' aşamasından.
"""
import json
import math
from pathlib import Path

import numpy as np
import polars as pl

import radar as R

ANALYSIS_ROOT = Path("reports/analysis")
SAMPLES_ROOT = Path("samples/analysis")      # src/sample_pack.py: anonim örnek analizler (temiz kurulumda da açılır)
JURY_FILE = Path("data/reference/v1/jury_set.json")
SMOKE_RADIUS = 144.0          # birim (CS2 duman yarıçapı yaklaşık)
INFERNO_RADIUS = 120.0
AIM_LENGTH = 400.0            # bakış yönü oku uzunluğu (birim)


def jury_roles() -> dict:
    if not JURY_FILE.exists():
        return {}
    return {e["match"]: e for e in json.loads(JURY_FILE.read_text(encoding="utf-8"))["examples"] if e.get("match")}


def list_analyses(root: Path | None = None) -> list[dict]:
    """Arayüzde açılabilecek analizler (skoru olanlar). Ham dosya adı gösterilmez; kimlik = demo_id.
    root verilmezse reports/analysis/ (yerel, git dışı) + samples/analysis/ (anonim örnek paketi, git'te);
    aynı demo_id iki yerde varsa yerel olan kullanılır."""
    out, roles, seen = [], jury_roles(), set()
    roots = [Path(root)] if root is not None else [ANALYSIS_ROOT, SAMPLES_ROOT]
    for w in [w for r in roots for w in sorted(r.glob("*/metadata.json"))]:
        work = w.parent
        meta = json.loads(w.read_text(encoding="utf-8"))
        if meta["demo_id"] in seen:
            continue
        seen.add(meta["demo_id"])
        oof = work / "oof_result.json"
        if oof.exists():
            r = json.loads(oof.read_text(encoding="utf-8"))
            role = roles.get(r["match"])
            out.append(dict(work=work, demo_id=meta["demo_id"], map=meta["map"], mode="cs2cd_oof", training=True,
                            match=r["match"], jury=role,
                            title=f"CS2CD {r['match']} · {meta['map']} · eğitim verisinden"
                                  + (f" · örnek: {role['label']}" if role else "")))
        elif (work / "player_scores.parquet").exists():
            out.append(dict(work=work, demo_id=meta["demo_id"], map=meta["map"], mode="faz2", training=False,
                            match=None, jury=None, source=meta.get("source"),
                            title=f"{meta['map']} · {meta.get('source')} · {meta['demo_id']}"
                                  + (f" · {SOURCE_ROLE[meta.get('source')]}" if meta.get("source") in SOURCE_ROLE else "")))
    return out


# HLTV profesyonel demoları kullanıcıya "model sınırlılığı örneği" olarak gösterilir (açılış demosu değil)
SOURCE_ROLE = {"hltv": "model sınırlılığı örneği"}


def distribution(work: Path) -> dict | None:
    """demo düzeyi dağılım bilgisi (yalnız gösterim; skorları/önceliği değiştirmez). Dondurulmuş referans
    (data/reference/v1/distribution_reference.json) + qc.json + features.parquet ile anında hesaplanır."""
    import distribution_check as DC
    if not DC.REF_FILE.exists() or not (Path(work) / "qc.json").exists():
        return None
    return DC.check_analysis(Path(work))


NOT_DEFAULT_SOURCES = {"hltv"}   # profesyonel maçlar varsayılan açılış demosu olmaz (yüksek beceri yanlış alarm riski)


def default_index(entries: list[dict], want: str | None = None) -> int:
    """Açılışta seçili analiz: ?demo=<id> varsa o; yoksa HLTV olmayan ilk analiz; hiçbiri yoksa 0."""
    for i, e in enumerate(entries):
        if want and e["demo_id"] == want:
            return i
    return next((i for i, e in enumerate(entries) if e.get("source") not in NOT_DEFAULT_SOURCES), 0)


def load(entry: dict) -> dict:
    work = Path(entry["work"])
    pre = "oof_" if entry["mode"] == "cs2cd_oof" else ""
    d = dict(entry=entry, work=work,
             meta=json.loads((work / "metadata.json").read_text(encoding="utf-8")),
             qc=json.loads((work / "qc.json").read_text(encoding="utf-8")),
             players=pl.read_parquet(work / f"{pre}player_scores.parquet"),
             events=pl.read_parquet(work / f"{pre}event_scores.parquet"),
             map_events=pl.read_parquet(work / f"{pre}map_events.parquet"),
             kill_windows=pl.read_parquet(work / "kill_windows.parquet"),
             utility=pl.read_parquet(work / "utility.parquet"),
             labels=pl.read_parquet(work / "dataset_labels.parquet") if pre and (work / "dataset_labels.parquet").exists() else None,
             result=json.loads((work / (f"{pre}result.json" if pre else "model_result.json")).read_text(encoding="utf-8")))
    d["radar"] = radar_info(work, d["meta"]["map"])
    return d


def radar_info(work: Path, map_name: str, radar_dir: Path = R.RADAR_DIR) -> dict | None:
    """Radar bilgisi; dosyalar yoksa çökme yok (README: Valve dosyaları dağıtılmaz, kullanıcı kendi CS2'sinden üretir).
    JSON var + PNG yok → aynı dönüşüm, görüntüsüz. JSON yok → konumların sınırlarından boş zemin (R.fallback_info)."""
    info = R.load_info(map_name, radar_dir)
    if info is not None:
        img = info["levels"][0].get("image")
        if img and (Path(radar_dir) / img).exists():
            return info
        return info | dict(levels=[lv | {"image": None} for lv in info["levels"]], image_missing=True)
    f = Path(work) / "map_context.parquet"
    if not f.exists():
        return None
    xy = pl.read_parquet(f, columns=["X", "Y"]).drop_nulls()
    return R.fallback_info(xy["X"].to_list(), xy["Y"].to_list())



def context_round(work: Path, rnd: int) -> pl.DataFrame:
    """Bir round'un 8 Hz konumları (yalnız o round okunur)."""
    return (pl.scan_parquet(Path(work) / "map_context.parquet").filter(pl.col("round") == rnd).collect()
              .sort("tick", "player"))


def positions_at(ctx: pl.DataFrame, tick: int) -> pl.DataFrame:
    """tick'e en yakın (<=) örnekteki bütün oyuncular."""
    ts = ctx.filter(pl.col("tick") <= tick)["tick"]
    t = ts.max() if ts.len() else ctx["tick"].min()
    return ctx.filter(pl.col("tick") == t)


def live_trails(ctx: pl.DataFrame, tick: int) -> dict[str, pl.DataFrame]:
    """Her oyuncunun round başından tick'e kadar YALNIZ 'live' aşamasındaki, hayattayken yolu."""
    c = ctx.filter((pl.col("tick") <= tick) & pl.col("is_alive"))
    if "phase" in c.columns:
        c = c.filter(pl.col("phase") == "live")
    return {p: g.sort("tick") for (p,), g in c.group_by("player", maintain_order=True)}


def kill_window(kw: pl.DataFrame, kill_id: int, role: str) -> pl.DataFrame:
    return kw.filter((pl.col("kill_id") == kill_id) & (pl.col("role") == role)).sort("tick").drop_nulls("X")


def aim_endpoint(x: float, y: float, yaw: float, length: float = AIM_LENGTH) -> tuple[float, float]:
    """Oyun koordinatında bakış yönü: yaw 0 = +X (doğu), 90 = +Y (kuzey)."""
    r = math.radians(yaw)
    return x + length * math.cos(r), y + length * math.sin(r)


UTILITY_MAX_S = {"smoke": 25.0, "molotof": 8.0}   # oyun içi ömür üst sınırı (ölçüm: medyan ~22 sn / ~7 sn)
_UTIL_EVENTS = (("smokegrenade_detonate", "smokegrenade_expired", "smoke"), ("inferno_startburn", "inferno_expire", "molotof"))


def utility_intervals(util: pl.DataFrame, tickrate: int = 64) -> pl.DataFrame:
    """Duman/molotof aralıkları (yalnız gösterim): her başlangıç, aynı entityid'nin KENDİSİNDEN
    SONRAKİ İLK bitiş olayıyla eşlenir (oyun entityid'yi sonraki round'larda yeniden kullanır; eski birleştirme sonraki
    round'ların bitişleriyle de eşleşip dumanı round'lar boyunca 'etkin' gösteriyordu). Bitiş yoksa ya da ömürden
    uzunsa süre UTILITY_MAX_S ile sınırlanır."""
    parts = []
    for start, end, kind in _UTIL_EVENTS:
        s_ = util.filter(pl.col("event") == start).select("tick", "entityid", "x", "y", "z").sort("tick")
        e_ = util.filter(pl.col("event") == end).select("entityid", end_tick="tick").sort("end_tick")
        cap = int(UTILITY_MAX_S[kind] * tickrate)
        if s_.height == 0:
            continue
        j = (s_.join_asof(e_, left_on="tick", right_on="end_tick", by="entityid", strategy="forward", check_sortedness=False)
               .with_columns(end_tick=pl.min_horizontal(pl.col("end_tick").fill_null(pl.col("tick") + cap),
                                                        pl.col("tick") + cap))
               .with_columns(kind=pl.lit(kind)))
        parts.append(j.select("x", "y", "z", "tick", "end_tick", "kind"))
    return pl.concat(parts) if parts else pl.DataFrame(schema={"x": pl.Float64, "y": pl.Float64, "z": pl.Float64,
                                                               "tick": pl.Int64, "end_tick": pl.Int64, "kind": pl.Utf8})


def active_utility(util: pl.DataFrame, tick: int) -> pl.DataFrame:
    """tick anında etkin dumanlar ve molotoflar (başlangıç ≤ tick < bitiş; eşleme utility_intervals ile)."""
    iv = utility_intervals(util)
    return iv.filter((pl.col("tick") <= tick) & (pl.col("end_tick") > tick)) if iv.height else iv


def player_rows(d: dict) -> pl.DataFrame:
    """Oyuncu tablosu (model çıktısı). Veri seti etiketi AYRI tabloda (dataset_label_rows)."""
    p = d["players"]
    cols = {"player": "oyuncu", "style": "stil", "analyzed_kills": "öldürme", "player_raw_score": "model skoru",
            "style_percentile": "stil içi yüzdelik", "review_priority": "inceleme önceliği",
            "review_points": "inceleme noktası", "low_confidence_style": "düşük güven (sniper)",
            "g0_above_threshold": "G0 (karşılaştırma)"}
    have = [c for c in cols if c in p.columns]
    return p.select(have).rename({c: cols[c] for c in have})


# ---------------------------------------------------------------- arayüz biçimlendirme (yalnız gösterim; skor değişmez)
PRIORITY_ORDER = ["yüksek inceleme önceliği", "orta inceleme önceliği", "düşük inceleme önceliği", "yetersiz veri"]
UNKNOWN = "bilinmiyor"


def _missing(v) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v))


def priority_rank(p) -> int:
    """yüksek < orta < düşük < yetersiz veri < diğer (ör. 'değerlendirme dışı')."""
    return PRIORITY_ORDER.index(p) if p in PRIORITY_ORDER else len(PRIORITY_ORDER)


def player_order(players: pl.DataFrame) -> list[str]:
    """Oyuncu listesi sırası: önce inceleme önceliği, sonra stil içi yüzdelik (büyükten küçüğe), sonra ad."""
    rows = players.select("player", "review_priority", "style_percentile").to_dicts()
    rows.sort(key=lambda r: (priority_rank(r["review_priority"]),
                             -(r["style_percentile"] if not _missing(r["style_percentile"]) else -1.0),
                             int(r["player"].split("_")[-1]) if r["player"].split("_")[-1].isdigit() else 0))
    return [r["player"] for r in rows]


def fmt_value(v, nd: int = 0, unit: str = "") -> str:
    """Sayı → metin; NaN/None → 'bilinmiyor' (boş hücre yerine)."""
    if _missing(v):
        return UNKNOWN
    return f"{v:.{nd}f}{unit}"


def fmt_pct(v, nd: int = 1) -> str:
    """Yüzdelik → '%99.7' (Türkçe yazım); NaN/None → 'bilinmiyor'."""
    return UNKNOWN if _missing(v) else f"%{v:.{nd}f}"


def yes_no(v) -> str:
    """Bayrak → 'evet' / 'hayır' / 'bilinmiyor' (onay kutusu yerine)."""
    if _missing(v):
        return UNKNOWN
    return "evet" if bool(v) else "hayır"


def visibility_text(v) -> str:
    return UNKNOWN if _missing(v) else ("görünür" if v else "görünmez")


def explanation_bullets(top_features_json, fallback: str | None = None, n: int = 3) -> list[tuple[str, str]]:
    """Açıklama → en fazla n kısa madde: (özellik ve değer, temiz dağılıma göre karşılaştırma).
    Kaynak Faz 2'nin top_features JSON'u; yoksa '; ' ile ayrılmış açıklama metni."""
    texts = []
    if top_features_json:
        try:
            texts = [f["text"] for f in json.loads(top_features_json)]
        except (ValueError, KeyError, TypeError):
            texts = []
    if not texts and fallback:
        texts = fallback.split("; ")
    out = []
    for t in texts[:n]:
        head, _, tail = t.partition(" — ")
        out.append((head.strip(), tail.strip()))
    return out


def detail_tables(d: dict, player: str) -> dict[str, pl.DataFrame]:
    """'Detaylı veriler' bölümünün tabloları; eksik değerler 'bilinmiyor', bayraklar 'evet/hayır'."""
    p = player_rows(d)
    ev = d["events"].filter(pl.col("attacker") == player).sort("event_score", descending=True)
    events = pl.DataFrame(dict(
        olay=ev["kill_id"].to_list(), round=ev["round"].to_list(), tick=ev["kill_tick"].to_list(),
        silah=ev["weapon"].to_list(), kurban=ev["victim"].to_list(),
        **{"olay skoru": [fmt_value(v, 3) for v in ev["event_score"]],
           "stil içi olay yüzdeliği": [fmt_pct(v) for v in ev["event_style_percentile"]],
           "inceleme noktası": [yes_no(v) for v in ev["review_point"]],
           "wallbang": [yes_no(None if _missing(v) else v > 0) for v in ev["wallbang"]],
           "duman": [yes_no(v) for v in ev["thrusmoke"]],
           "kurban görünür": [visibility_text(v) for v in ev["victim_visible_at_kill"]],
           "tepki (ms)": [fmt_value(v, 0) for v in ev["reaction_ms"]],
           "pre-FOV (°)": [fmt_value(v, 1) for v in ev["prefov_angle"]]}))
    def cell(c, v):
        if c == "stil içi yüzdelik":
            return fmt_pct(v)
        if isinstance(v, bool):
            return yes_no(v)
        if isinstance(v, float):
            return fmt_value(v, 3)
        return UNKNOWN if v is None else str(v)
    players = pl.DataFrame({c: [cell(c, v) for v in p[c]] for c in p.columns}, schema={c: pl.String for c in p.columns})
    return dict(players=players, events=events)


# ---------------------------------------------------------------- metrik kartları ("neden bu öncelik?")
FEATURE_SHORT = {   # arayüz için kısa adlar (açıklama metninin uzun adları score_demo.FEATURE_TEXT'te)
    "speed_mean": "Nişan hızı (1 sn)", "speed_max": "En yüksek nişan hızı", "speed_last_mean": "Son 250 ms nişan hızı",
    "jitter": "Nişan titrekliği", "still_ratio": "Hareketsiz nişan oranı", "pitch_range_last": "Dikey nişan aralığı",
    "shots": "Atış sayısı (1 sn)", "after_move": "Öldürme sonrası dönüş", "inv_angle_med": "Görünmezken nişan açısı",
    "inv_frac_lt10": "Görünmeze <10° nişan oranı", "prefov_angle": "Pre-aim açısı", "reaction_ms": "Tepki süresi",
}
FEATURE_GROUP = {"reaction_ms": "Tepki", "inv_angle_med": "Görünürlük / pre-aim", "inv_frac_lt10": "Görünürlük / pre-aim",
                 "prefov_angle": "Görünürlük / pre-aim"}
LOW_DATA_KILLS = 10       # yalnız gösterim notu ("az veri"); skor kuralı (MIN_KILLS = 3) değişmez


def metric_cards(top_features_json, n: int = 3) -> list[dict]:
    """Faz 2 top_features → kısa metrik kartları: grup, kısa ad, değer, temizlerle karşılaştırma, yüzdelik."""
    import re
    from score_demo import FEATURE_TEXT
    try:
        feats = json.loads(top_features_json) if top_features_json else []
    except (ValueError, TypeError):
        feats = []
    out = []
    for f in feats[:n]:
        name, unit, form = FEATURE_TEXT.get(f["feature"], (f["feature"], "", "{:.2f}"))[:3]
        m = re.search(r"%\d+'\S+ \S+", f.get("text", ""))
        out.append(dict(feature=f["feature"], group=FEATURE_GROUP.get(f["feature"], "Nişan davranışı"),
                        name=FEATURE_SHORT.get(f["feature"], name),
                        value=UNKNOWN if _missing(f.get("value")) else form.format(f["value"]) + (f" {unit}" if unit else ""),
                        compare=("temizlerin " + m.group(0)) if m else "", pct=f.get("notable_pct"),
                        typical=(f.get("notable_pct") or 0) < 75))
    return out


def reason_lines(cards: list[dict], n: int = 3) -> list[dict]:
    """'Neden bu öncelik?' için en önemli n kısa neden (yalnız gösterim): metric_cards çıktısının ilk n'i,
    yalnız ad + değer + karşılaştırma. Sıra ve değerler aynı; yeni hesap yok. Grup ve 'tipik aralıkta' ayrıntıda."""
    return [dict(name=c["name"], value=c["value"], compare=c["compare"], typical=c["typical"], pct=c["pct"])
            for c in cards[:n]]


def live_rounds(meta: dict) -> list[dict]:
    """Round tablosundan yalnız canlı oyunu olan round'lar. Son satır maç bitişinden sonraki yarım 'round'dur
    (donma süresi var, canlı oyun yok) → zaman çizelgesinde R(N+1) gibi görünmesin."""
    return [r for r in meta.get("rounds_table") or [] if r.get("round", 0) > 0 and r.get("live_start_tick") is not None]


def timeline_pick(points: list[dict] | None, events: pl.DataFrame) -> dict | None:
    """Zaman çizelgesinde tıklanan nokta → seçilecek olay. YALNIZ kill_id'ye (noktanın customdata'sı) dayanır;
    point_index / iz sırası kullanılmaz → olay filtreleri sırayı değiştirse de doğru olay seçilir.
    Dönüş: {kill_id, player (saldırgan), round, tick} ya da None (boş seçim / bilinmeyen kill_id)."""
    for pt in points or []:
        cd = pt.get("customdata")
        if isinstance(cd, (list, tuple)):
            cd = cd[0] if cd else None
        if cd is None:
            continue
        try:
            kid = int(cd)
        except (TypeError, ValueError):
            continue
        row = events.filter(pl.col("kill_id") == kid)
        if row.height:
            r = row.row(0, named=True)
            return dict(kill_id=kid, player=r["attacker"], round=r["round"], tick=r["kill_tick"])
    return None


# ---------------------------------------------------------------- olay filtreleri (Faz 3.5)
# YALNIZ GÖSTERİM / İNCELEME: hangi olayların listelendiğini daraltır; skor, öncelik, eşik değişmez.
FILTERS = {
    "review_point": ("İnceleme noktası", "Olay skoru stil içi olay eşiğinin (temizlerin %95'lik dilimi) üstünde."),
    "wallbang": ("Wallbang", "Mermi duvar/obje delerek isabet etti."),
    "smoke": ("Duman içinden", "Öldürme duman içinden yapıldı."),
    "not_visible": ("Öldürme anında görünür değil",
                    "Ray-cast'e göre kurban öldürme anında görünür değildi (duvar, duman vb.). Görünür olmamak otomatik "
                    "olarak wallhack anlamına gelmez (wallbang, duman, ses/bilgiyle atış olabilir). 'Bilinmiyor' olanlar "
                    "eşleşmez."),
    "aim_near_hidden": ("Görünmez hedefe nişan yakınlığı yüksek",
                        "Temiz referanstaki olaylara göre uç nişan yakınlığıdır; tek başına hile göstergesi değildir. "
                        "Tanım: kurban görünmezken nişanın kurbana medyan açısı, aynı silah sınıfındaki temiz eğitim "
                        "öldürmelerinin en küçük %10'unda. Model eşiği değildir, yalnız inceleme filtresidir."),
}
AIM_NEAR_Q = 0.10
SCORE_PCT_STEPS = [0, 75, 90, 95, 99]          # 0 = kapalı


def aim_near_thresholds(ref_file: Path = Path("data/reference/v1/clean_kill_features.parquet")) -> dict:
    """Silah sınıfı başına temiz eğitim öldürmelerinde inv_angle_med'in %10'luk dilimi (dondurulmuş referanstan)."""
    ref = pl.read_parquet(ref_file, columns=["weapon_class", "inv_angle_med"]).drop_nulls()
    return {wc: float(g["inv_angle_med"].quantile(AIM_NEAR_Q, "linear")) for (wc,), g in ref.group_by("weapon_class")}


def filter_events(events: pl.DataFrame, active=(), min_pct: float = 0, near_thr: dict | None = None) -> pl.DataFrame:
    """Seçili filtrelerin hepsini (VE) sağlayan olaylar; satır sırası korunur. Eksik değer hiçbir filtreyle eşleşmez."""
    conds = []
    if "review_point" in active:
        conds.append(pl.col("review_point").fill_null(False))
    if "wallbang" in active:
        conds.append(pl.col("wallbang").fill_null(0) > 0)
    if "smoke" in active:
        conds.append(pl.col("thrusmoke").fill_null(False))
    if "not_visible" in active:
        conds.append(pl.col("victim_visible_at_kill").eq(False).fill_null(False))
    if "aim_near_hidden" in active:
        thr = near_thr if near_thr is not None else aim_near_thresholds()
        conds.append((pl.col("inv_angle_med") <= pl.col("weapon_class").replace_strict(thr, default=None,
                                                                                         return_dtype=pl.Float64))
                     .fill_null(False))
    if min_pct and min_pct > 0:
        conds.append(pl.col("event_style_percentile").fill_null(-1) >= min_pct)
    return events.filter(pl.all_horizontal(conds)) if conds else events


def filter_summary(active, min_pct: float, n: int, total: int) -> str:
    parts = [FILTERS[k][0] for k in FILTERS if k in active] + ([f"stil içi olay yüzdeliği ≥ %{min_pct:g}"] if min_pct else [])
    return ("Filtre: " + " + ".join(parts) if parts else "Filtre yok") + f" · {n} / {total} olay"


# ---------------------------------------------------------------- radar etiketleri (Faz 3.5)
LABEL_MIN_PX = 30.0      # radar pikseli (1024'lük ızgara); bundan yakın etiketler birleştirilir


def _natural(name: str) -> tuple:
    """Player_2 < Player_10 (sayıya göre sıralama)."""
    head, _, num = str(name).rpartition("_")
    return (head, int(num)) if num.isdigit() else (str(name), -1)


def declutter_labels(xs, ys, names, reserved=(), min_px: float = LABEL_MIN_PX) -> tuple[list[str], list[int]]:
    """Yakın oyuncu etiketlerini sadeleştirir; NOKTALAR değişmez, yalnız yazı. Ad sırasına göre (deterministik) gezilir:
    bir oyuncu, etiketi gösterilen başka bir noktaya (veya `reserved` = ayrıca etiketlenen seçili oyuncu/kurban)
    min_px'ten yakınsa etiketi yazılmaz, o etiket "+n" alır. Dönüş: (nokta başına metin, reserved başına gizlenen sayısı).
    Kimlik kaybolmaz: arayüz her noktada hover'da tam Player_N adını gösterir."""
    anchors = [(float(x), float(y), None) for x, y in reserved]          # (x, y, nokta indeksi | None)
    count_pt, count_res = {}, [0] * len(anchors)
    texts = [""] * len(names)
    for i in sorted(range(len(names)), key=lambda k: _natural(names[k])):
        x, y = float(xs[i]), float(ys[i])
        near = next((j for j, (ax, ay, _) in enumerate(anchors) if (ax - x) ** 2 + (ay - y) ** 2 < min_px ** 2), None)
        if near is None:
            anchors.append((x, y, i))
            count_pt[i] = 0
        elif anchors[near][2] is None:
            count_res[near] += 1
        else:
            count_pt[anchors[near][2]] += 1
    for i, c in count_pt.items():
        texts[i] = names[i] + (f" +{c}" if c else "")
    return texts, count_res


# ---------------------------------------------------------------- oyuncu karşılaştırma (Faz 3.5)
# YALNIZ GÖSTERİM: mevcut model çıktıları + özellik özetleri yan yana. Yeni skor, eşik, sıralama veya hüküm YOK.
# Özet = oyuncunun modele giren BÜTÜN öldürmeleri üzerinden ortalama (olay filtreleri uygulanmaz; "Neden bu öncelik?"
# kartlarıyla aynı tanım). Yüzdelik = aynı stildeki temiz eğitim oyuncularının oyuncu ortalamalarının yüzde kaçı bu
# değerden KÜÇÜK (data/reference/v1/clean_player_features.parquet; score_demo.percentile_below ile aynı).
COMPARE_FEATURES = [   # (grup, özellik, etiket, birim, biçim, anlam notu)
    ("Tepki", "reaction_ms", "Ortalama tepki süresi (görünür olma → ilk isabet)", "ms", "{:.0f}",
     "Düşük = daha kısa tepki"),
    ("Nişan", "speed_max", "En yüksek nişan hızı (son 1 sn)", "°/s", "{:.0f}", None),
    ("Nişan", "jitter", "Nişan titrekliği", "", "{:.2f}", None),
    ("Nişan", "still_ratio", "Hareketsiz nişan oranı (son 1 sn)", "", "{:.2f}", None),
    ("Görünürlük / ön-nişan", "prefov_angle", "Pre-FOV açısı (rakip görünmeden ~190 ms önce)", "°", "{:.1f}",
     "Düşük = hedefe daha yakın nişan"),
    ("Görünürlük / ön-nişan", "inv_angle_med", "Görünmezken nişan açısı (medyan)", "°", "{:.1f}",
     "Düşük = hedefe daha yakın nişan"),
    ("Görünürlük / ön-nişan", "inv_frac_lt10", "Görünmez rakibe 10 dereceden yakın nişan oranı", "", "{:.2f}",
     "Yüksek = görünmezken nişan daha sık hedefe yakın"),
]
COMPARE_EXTRA_AIM = [
    ("Nişan (ek)", "speed_mean", "Ortalama nişan hızı (son 1 sn)", "°/s", "{:.0f}", None),
    ("Nişan (ek)", "speed_last_mean", "Nişan hızı (son 250 ms)", "°/s", "{:.0f}", None),
    ("Nişan (ek)", "pitch_range_last", "Dikey nişan aralığı (son 250 ms)", "°", "{:.1f}", None),
    ("Nişan (ek)", "shots", "Atış sayısı (son 1 sn)", "", "{:.1f}", None),
    ("Nişan (ek)", "after_move", "Öldürme sonrası dönüş (500 ms)", "°", "{:.1f}", None),
]
COMPARE_NOTE = ("Bu ekran yalnız mevcut model çıktılarını ve özellik özetlerini yan yana gösterir; yeni skor üretmez, "
                "oyuncuları sıralamaz, hüküm vermez. Özetler modele giren bütün öldürmelerden hesaplanır (olay "
                "filtreleri uygulanmaz).")
_CLEAN_PLAYERS = None


def _clean_players() -> pl.DataFrame:
    global _CLEAN_PLAYERS
    if _CLEAN_PLAYERS is None:
        _CLEAN_PLAYERS = pl.read_parquet(Path("data/reference/v1/clean_player_features.parquet"))
    return _CLEAN_PLAYERS


def _pct_below(style, feature, v):
    """Aynı stildeki temiz oyuncu ortalamalarının yüzde kaçı v'den küçük (0–100); hesaplanamazsa None."""
    if style is None or feature is None or v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    ref = _clean_players().filter(pl.col("stil") == style)
    if feature not in ref.columns or ref.height == 0:
        return None
    arr = ref[feature].to_numpy().astype(float)
    arr = arr[~np.isnan(arr)]
    return round(float((arr < v).mean()) * 100, 1) if len(arr) else None


def _direction(fa: str, fb: str, va, vb) -> str:
    """Yalnız yön: gösterilen değerler eşitse 'eşit'; biri yoksa 'karşılaştırılamaz'. İyi/kötü anlamı YOK."""
    if va is None or vb is None:
        return "karşılaştırılamaz"
    if fa == fb:
        return "eşit"
    return "A daha yüksek" if va > vb else "B daha yüksek"


def compare_players(d: dict, a: str, b: str, extra: bool = False) -> list[dict]:
    """İki oyuncunun karşılaştırma satırları (yalnız mevcut çıktılar; modele giren bütün öldürmeler)."""
    P, ev = d["players"], d["events"]
    rows = []
    pr = {}
    for p in (a, b):
        f = P.filter(pl.col("player") == p)
        pr[p] = f.row(0, named=True) if f.height else {}
    st = {p: pr[p].get("style") if pr[p].get("player_raw_score") is not None else None for p in (a, b)}

    def model_row(key, label, fmt=None, text=False, pct=False, note=None):
        va, vb = pr[a].get(key), pr[b].get(key)
        if text:
            fa = str(va) if va is not None else UNKNOWN
            fb = str(vb) if vb is not None else UNKNOWN
            dirn = "aynı" if fa == fb else "farklı"
        else:
            fa = fmt.format(va) if va is not None else UNKNOWN
            fb = fmt.format(vb) if vb is not None else UNKNOWN
            dirn = _direction(fa, fb, va, vb)
        rows.append(dict(group="Model çıktısı", key=key, label=label, note=note, a=va, b=vb, a_text=fa, b_text=fb,
                         a_pct=va if pct else None, b_pct=vb if pct else None, direction=dirn, kind="model"))

    model_row("player_raw_score", "Model skoru (ortalama)", "{:.3f}")
    model_row("review_priority", "İnceleme önceliği", text=True)
    model_row("style", "Stil", text=True)
    model_row("style_percentile", "Stil içi yüzdelik", "%{:.1f}", pct=True,
              note="Model skorunun aynı stildeki temiz oyunculara göre konumu (mevcut G1 çıktısı)")
    model_row("analyzed_kills", "Modele giren öldürme", "{:.0f}")
    model_row("review_points", "İnceleme noktası", "{:.0f}")

    for group, f, label, unit, fmt, note in COMPARE_FEATURES + (COMPARE_EXTRA_AIM if extra else []):
        vals = {}
        for p in (a, b):
            s = ev.filter(pl.col("attacker") == p)[f].cast(pl.Float64).drop_nulls().drop_nans()
            vals[p] = (float(s.mean()) if s.len() else None, s.len())
        (va, na), (vb, nb) = vals[a], vals[b]
        u = f" {unit}" if unit else ""
        fa = fmt.format(va) + u if va is not None else UNKNOWN
        fb = fmt.format(vb) + u if vb is not None else UNKNOWN
        rows.append(dict(group=group, key=f, label=label, note=note, a=va, b=vb, a_text=fa, b_text=fb,
                         a_n=na, b_n=nb, a_pct=_pct_below(st[a], f, va), b_pct=_pct_below(st[b], f, vb),
                         direction=_direction(fa, fb, va, vb), kind="feature"))

    def count_row(group, key, label, cond, ref_feature=None, note=None):
        out = {}
        for p in (a, b):
            e = ev.filter(pl.col("attacker") == p).select(c=cond)
            known = e.filter(pl.col("c").is_not_null()).height
            hit = e.filter(pl.col("c").fill_null(False)).height
            out[p] = (hit, known, hit / known if known else None)
        (ha, ka, ra), (hb, kb, rb) = out[a], out[b]
        fa = f"{ha} / {ka} (%{ra * 100:.0f})" if ra is not None else UNKNOWN
        fb = f"{hb} / {kb} (%{rb * 100:.0f})" if rb is not None else UNKNOWN
        rows.append(dict(group=group, key=key, label=label, note=note, a=ra, b=rb, a_text=fa, b_text=fb,
                         a_n=ka, b_n=kb, a_count=ha, b_count=hb,
                         a_pct=_pct_below(st[a], ref_feature, ra), b_pct=_pct_below(st[b], ref_feature, rb),
                         direction=_direction(f"{ra:.3f}" if ra is not None else "",
                                              f"{rb:.3f}" if rb is not None else "", ra, rb), kind="count"))

    count_row("Görünürlük / ön-nişan", "not_visible", "Öldürme anında görünür olmayan öldürme",
              pl.col("victim_visible_at_kill").not_(),
              note="Görünür olmamak otomatik olarak wallhack anlamına gelmez (wallbang, duman, ses/bilgi)")
    count_row("Wallbang / duman", "wallbang", "Wallbang öldürme", pl.col("wallbang") > 0, ref_feature="wallbang")
    count_row("Wallbang / duman", "thrusmoke", "Duman içinden öldürme", pl.col("thrusmoke"))
    return rows


# ---------------------------------------------------------------- ortak metinler: arayüz + HTML raporu
LOW_CONF = "Sniper ağırlıklı stilde ayırt etme gücü düşük (stil içi AUC ~0.75): bu oyuncular için güven düşük."
SNIPER_LIMIT = "Bilinen sınırlılık: sniper ağırlıklı oyuncularda yanlış alarm riski."
RADAR_MISSING = ("Harita görseli bulunamadı (data/maps/radar/): konumlar ve yollar boş zemin üzerinde gösteriliyor. "
                 "Radar görselleri Valve dosyasıdır, projeyle dağıtılmaz; kendi CS2 kurulumunuzdan üretmek için README "
                 "'Harita dosyaları' bölümüne bakın.")
PRO_FP_LIMIT = ("Professional/high-skill false-positive risk: Model v1, çok yüksek becerili oyuncularda (ör. profesyonel maçlar; "
                "özellikle kısa tepki süresi) beceriyi hile benzeri davranıştan her zaman ayırt edemez ve yüksek "
                "inceleme önceliği verebilir. Sistem CS2CD benzeri matchmaking maçları için kuruldu; demo düzeyi "
                "dağılım bilgisi bu riskin işaretidir.")
POINT_DEF = ("İnceleme noktası = tek bir öldürmenin model skoru, aynı stildeki temiz oyuncuların öldürmelerinin "
             "%95'lik diliminden (olay eşiği) yüksek. İnceleme önceliği ise oyuncunun ORTALAMA skorunu stil içi temiz "
             "oyuncu ortalamalarıyla karşılaştırır (oyuncu eşiği). Ortalamalar daha az saçıldığı için oyuncu eşiği "
             "daha düşüktür: yüksek öncelikli bir oyuncunun inceleme noktası 0 olabilir.")


def limitations(mode: str, result: dict, bold: bool = False) -> list[str]:
    """'Model sınırlılıkları' listesi — arayüz ve rapor AYNI metni kullanır."""
    from score_demo import DISCLAIMER, EVENT_NOTE
    head = f"**{DISCLAIMER}**" if bold else DISCLAIMER
    lim = [f"{head} Güncel demolarda kalibrasyon doğrulanmadı. Bu arayüz kimse hakkında hüküm vermez; "
           "yalnızca inceleme önceliği gösterir.",
           EVENT_NOTE, POINT_DEF,
           "İnceleme önceliği stil içi karşılaştırmadır (G1): oyuncu, aynı stildeki temiz oyuncuların skor dağılımına "
           "göre sıralanır. G0 (tek global eşik) yalnız karşılaştırma içindir (Detaylı veriler).",
           LOW_CONF, SNIPER_LIMIT,
           f"Az veri: {LOW_DATA_KILLS}'dan az öldürmesi olan oyuncuların ortalaması daha oynaktır (yalnız gösterim "
           "notu; skor kuralı ≥ 3 öldürme değişmedi).",
           "Görünürlük ray-cast ile hesaplanır (doğrulamada ~%96–97 uyum); 'bilinmiyor' = hesaplanamadı (ör. geçiş yok).",
           "Çok katlı haritalarda yalnız ana kat gösterilir.",
           PRO_FP_LIMIT]
    if mode == "cs2cd_oof":
        lim.insert(1, f"{result['mode_note']}: bu maç modelin eğitim havuzunda; skorlar maçı görmemiş kat "
                      "modellerinden (OOF). Veri seti etiketi model çıktısından ayrı gösterilir.")
    return lim


# ---------------------------------------------------------------- analiz geçmişi (Faz 3.5) — yalnız mevcut dosyalar
SAMPLE_DATE = "örnek paket"
DIST_TEXT = {0: "referans aralığında", 1: "dağılım farkı gözlendi", 2: "farklı"}


def _is_sample(work: Path) -> bool:
    try:
        Path(work).resolve().relative_to(SAMPLES_ROOT.resolve())
        return True
    except ValueError:
        return False


def history_rows(entries: list[dict]) -> list[dict]:
    """Analiz geçmişi satırları. Tarih = metadata.json dosya tarihi ("dosya tarihi"); örnek paketteki analizlerde
    tarih GÖSTERİLMEZ ("örnek paket": klonlama tarihi yanıltıcı olur). Sayılar player_scores'tan (mevcut çıktı)."""
    from datetime import datetime
    rows = []
    for e in entries:
        work = Path(e["work"])
        meta = json.loads((work / "metadata.json").read_text(encoding="utf-8"))
        pre = "oof_" if e["mode"] == "cs2cd_oof" else ""
        P = pl.read_parquet(work / f"{pre}player_scores.parquet", columns=["review_priority"])
        counts = {k: P.filter(pl.col("review_priority") == k).height for k in PRIORITY_ORDER}
        sample = _is_sample(work)
        ts = None if sample else (work / "metadata.json").stat().st_mtime
        role = []
        if sample:
            role.append("örnek paket")
        if e.get("jury"):
            role.append("örnek: " + e["jury"]["label"])
        if e.get("source") in SOURCE_ROLE:
            role.append(SOURCE_ROLE[e["source"]])
        dist = distribution(work)
        rows.append(dict(
            demo_id=e["demo_id"], work=str(work),
            source=("CS2CD (eğitim havuzu)" if e["mode"] == "cs2cd_oof" else (e.get("source") or meta.get("source")
                                                                                or UNKNOWN)),
            map=meta.get("map"), rounds=meta.get("rounds"),
            date=SAMPLE_DATE if sample else datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M"), timestamp=ts,
            high=counts["yüksek inceleme önceliği"], mid=counts["orta inceleme önceliği"],
            low=counts["düşük inceleme önceliği"], insufficient=counts["yetersiz veri"],
            role=" · ".join(role), dist=DIST_TEXT.get(dist["level"]) if dist else UNKNOWN))
    # yeniden eskiye; örnek paket en sonda (tarihi yok), kendi içinde demo_id sırası
    return sorted(rows, key=lambda r: (r["timestamp"] is None, -(r["timestamp"] or 0), r["demo_id"]))


# ---------------------------------------------------------------- çok katlı haritalar (Faz 3.5) — YALNIZ GÖSTERİM
# Ölçüm (README.md → Özellikler): Nuke ve Vertigo'da Valve kat eşiği konumlarla tutarlı; ±32 birim histerezis
# "1 sn içinde geri dönen" kat değişimini %14 → ~%2'ye indirir. Train'de eşik katları ayırmıyor (%97 "alt") → tek kat.
MULTILEVEL_MAPS = {"de_nuke", "de_vertigo"}
LEVEL_HYST = 32.0
LEVEL_LABEL = {"default": "Üst kat", "lower": "Alt kat"}
LEVEL_FROM_LABEL = {v: k for k, v in LEVEL_LABEL.items()}
SINGLE_LEVEL_NOTE = {"de_train": "Train için mevcut radar verisi katları güvenilir biçimde ayırmıyor; tek kat "
                                 "görünümü kullanılıyor."}


def level_enabled(info: dict | None, map_name: str) -> bool:
    """Kat seçimi yalnız ölçümle doğrulanan haritalarda ve iki kat görseli varken."""
    return bool(info) and map_name in MULTILEVEL_MAPS and info.get("multi_level") and all(
        lv.get("image") for lv in info["levels"])


def level_boundary(info: dict) -> float:
    return next(lv["z_min"] for lv in info["levels"] if lv["name"] == "default")


def assign_levels(zs, boundary: float, hyst: float = LEVEL_HYST) -> list[str]:
    """Zaman sıralı Z dizisi → kat dizisi (histerezis: sınırı ±hyst geçmeden kat değişmez; ilk örnek ham eşikle)."""
    out, lower = [], None
    for z in zs:
        if z is None or (isinstance(z, float) and math.isnan(z)):
            out.append(out[-1] if out else "default")
            continue
        if lower is None:
            lower = z < boundary
        elif z < boundary - hyst:
            lower = True
        elif z > boundary + hyst:
            lower = False
        out.append("lower" if lower else "default")
    return out


def player_level(ctx: pl.DataFrame, info: dict, player: str, tick: int, z_now: float | None = None) -> str:
    """Oyuncunun tick anındaki katı: round içindeki geçmişi (≤ tick) histerezisle izlenir; z_now verilirse son örnek."""
    h = ctx.filter((pl.col("player") == player) & (pl.col("tick") <= tick)).sort("tick")["Z"].to_list()
    if z_now is not None:
        h = h + [z_now]
    return assign_levels(h, level_boundary(info))[-1] if h else ("lower" if (z_now or 0) < level_boundary(info)
                                                                else "default")


def event_levels(ctx: pl.DataFrame, info: dict, e: dict) -> tuple[str, str]:
    """Seçili olayda saldırgan ve kurbanın katı (öldürme anındaki Z + round geçmişi, histerezisli)."""
    return (player_level(ctx, info, e["attacker"], e["kill_tick"], e.get("attacker_Z")),
            player_level(ctx, info, e["victim"], e["kill_tick"], e.get("victim_Z")))


def trail_segments(g: pl.DataFrame, info: dict, level: str) -> tuple[tuple[list, list], tuple[list, list]]:
    """Bir oyuncunun izi (X, Y, Z sıralı) → (seçili kattaki parçalar, diğer kattaki parçalar), None ile bölünmüş.
    Kat değişim noktası iki parçaya da eklenir (iz kopmasın)."""
    lv = assign_levels(g["Z"].to_list(), level_boundary(info))
    xs, ys = g["X"].to_list(), g["Y"].to_list()
    same, other = ([], []), ([], [])
    for i, (x, y, l) in enumerate(zip(xs, ys, lv)):
        tgt, alt = (same, other) if l == level else (other, same)
        if i > 0 and lv[i - 1] != l:
            alt[0].extend([x, None])
            alt[1].extend([y, None])
            tgt[0].append(xs[i - 1])
            tgt[1].append(ys[i - 1])
        tgt[0].append(x)
        tgt[1].append(y)
    return same, other
