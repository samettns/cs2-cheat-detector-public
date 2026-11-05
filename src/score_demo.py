"""Faz 2: Faz 1 çıktısı (15 özellik) → dondurulmuş Model B v1 → öldürme ve oyuncu skorları + açıklanabilir olay listesi.

Kullanım (tek komut; Faz 1 çıktısı yoksa önce Faz 1'i çalıştırır, varsa YENİDEN HESAPLAMAZ):
    .venv\\Scripts\\python.exe src\\score_demo.py <demo.dem> --source valve
    .venv\\Scripts\\python.exe src\\score_demo.py reports/analysis/<demo_id>

BU SKOR KALİBRE EDİLMİŞ HİLE OLASILIĞI DEĞİLDİR. Güncel demolarda kalibrasyon doğrulanmadı; çıktılar ham model skoru,
stil içi yüzdelik ve "inceleme önceliği" olarak verilir. Hiçbir oyuncu "hileci" olarak etiketlenmez.

Tanımlar (kod ve demo sonuçlarından önce yazıldı): MODEL_CARD.md (İnceleme önceliği).
  - Oyuncu skoru: öldürme skorlarının ortalaması (baseline.aggregate "mean"), en az 3 öldürme; azsa "yetersiz veri".
  - Varsayılan çıktı G1 mantığı = stil içi yüzdelik (models/v1/clean_reference.parquet). G0 yalnız karşılaştırma.
  - Eşik karşılaştırması değerlendirme koduyla aynı: skor > eşik (style_threshold.py, holdout.py).
  - İnceleme önceliği: yüksek = skor > G1 stil eşiği; orta = skor > stil içi temizlerin %90'lık dilimi; düşük = altı.
  - İnceleme noktası (olay): öldürme skoru > stil içi temiz OOF öldürme skorlarının %95'lik dilimi
    (data/reference/v1, src/explain_reference.py). Olay skorları ayrıca doğrulanmadı: "inceleme noktası"dır, kanıt değil.
  - Açıklama: stil (+ olayda silah sınıfı) içi temiz eğitim dağılımına göre yüzdelik; en belirleyici 3 özellik.

Faz 1'den okunanlar (hiçbiri değiştirilmez): qc.json, metadata.json, features.parquet, kills.parquet,
_stages/visibility_windows.parquet (görünürlük bağlamı; yoksa null). Yazılanlar: player_scores.parquet,
event_scores.parquet, map_events.parquet, model_result.json, model_report.md.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
from baseline import MIN_KILLS, aggregate          # noqa: E402
from style_threshold import MIN_CLEAN, STYLES, player_styles  # noqa: E402

DISCLAIMER = "Bu skor kalibre edilmiş hile olasılığı değildir."
EVENT_NOTE = ("Olay düzeyi skorlar ayrıca doğrulanmadı: model oyuncu etiketleriyle eğitildi; tek tek öldürme skorları "
              "'inceleme noktası' olarak kullanılabilir, 'hile anı kanıtı' olarak değil.")
SCORING_VERSION = "faz2-v1"
V1 = Path("models/v1")
REF = Path("data/reference/v1")
ID_COLS = ["kill_id", "match", "player", "tick", "weapon", "weapon_class"]
PHASE1_FILES = ["qc.json", "metadata.json", "features.parquet", "kills.parquet", "kill_windows.parquet",
                "map_context.parquet", "utility.parquet", "qc.md"]
PHASE2_FILES = ["player_scores.parquet", "event_scores.parquet", "map_events.parquet", "model_result.json",
                "model_report.md"]
ORTA_Q = 0.90                  # orta inceleme önceliği: stil içi temizlerin üst %10'u
TOP_EVENTS = 5                 # oyuncu başına listelenen en yüksek skorlu olay
MAP_TOP = 3                    # 2D harita için oyuncu başına (inceleme noktası olmasa da) en yüksek skorlu olay
TYPICAL_BELOW = 75             # açıklamada bu yüzdeliğin altı "(tipik aralıkta)" diye işaretlenir (yanıltmasın)
MIN_REF_KILLS = 100            # olay açıklamasında stil+silah referansı bundan azsa yalnız stil
STYLE_EN = {"sniper ağırlıklı": "sniper", "otomatik ağırlıklı": "automatic", "diğer": "other"}
WEAPON_TR = {"sniper": "sniper", "tabanca": "tabanca", "otomatik": "otomatik silah"}

# Açıklama etiketleri: (Türkçe ad, birim, biçim, düşük yön kelimesi, yüksek yön kelimesi)
FEATURE_TEXT = {
    "speed_mean": ("son 1 sn ortalama nişan hızı", "°/s", "{:.0f}", "düşük", "yüksek"),
    "speed_max": ("son 1 sn en yüksek nişan hızı", "°/s", "{:.0f}", "düşük", "yüksek"),
    "speed_last_mean": ("son 250 ms nişan hızı", "°/s", "{:.0f}", "düşük", "yüksek"),
    "jitter": ("nişan titrekliği", "", "{:.2f}", "düşük", "yüksek"),
    "still_ratio": ("son 1 sn hareketsiz nişan oranı", "", "{:.2f}", "düşük", "yüksek"),
    "pitch_range_last": ("son 250 ms dikey nişan aralığı", "°", "{:.1f}", "dar", "geniş"),
    "shots": ("son 1 sn atış sayısı", "", "{:.0f}", "az", "çok"),
    "after_move": ("öldürme sonrası 500 ms dönüş", "°", "{:.1f}", "az", "çok"),
    "inv_angle_med": ("rakip görünmezken nişan açısı (medyan)", "°", "{:.1f}", "küçük", "büyük"),
    "inv_frac_lt10": ("görünmez rakibe 10°'den yakın nişan oranı", "", "{:.2f}", "düşük", "yüksek"),
    "prefov_angle": ("rakip görünmeden ~190 ms önce nişan açısı", "°", "{:.1f}", "küçük", "büyük"),
    "reaction_ms": ("tepki süresi (görünür olma → ilk isabet)", "ms", "{:.0f}", "kısa", "uzun"),
}


class ScoringError(Exception):
    """Skor üretimini durduran, açıkça raporlanan hata (sessiz düzeltme yok)."""


# ---------------------------------------------------------------- Türkçe ek (yüzdelik ifadesi)
_POSS = {"1": "i", "2": "si", "3": "ü", "4": "ü", "5": "i", "6": "sı", "7": "si", "8": "i", "9": "u"}
_POSS_TENS = {"0": "ı", "10": "u", "20": "si", "30": "u", "40": "ı", "50": "si", "60": "ı", "70": "i", "80": "i",
              "90": "ı", "100": "ü"}


def pct_from(p: int) -> str:
    """'%98'inden', '%90'ından', '%62'sinden' (ayrılma eki, sayının okunuşuna göre)."""
    s = str(p)
    poss = _POSS_TENS[s] if (s in _POSS_TENS) else (_POSS[s[-1]] if s[-1] != "0" else _POSS_TENS[s[-2:] if len(s) > 1 else s])
    front = poss[-1] in "iü"
    return f"%{p}'{poss}n{'den' if front else 'dan'}"


# ---------------------------------------------------------------- yardımcılar
def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def model_identity() -> dict:
    files = [V1 / "model_B.joblib", V1 / "features.json", V1 / "thresholds.json", V1 / "clean_reference.parquet",
             REF / "reference.json", REF / "kill_oof_clean.parquet", REF / "clean_kill_features.parquet",
             REF / "clean_player_features.parquet"]
    return {str(p).replace("\\", "/"): sha256(p) for p in files}


def load_frozen() -> dict:
    meta = json.loads((V1 / "features.json").read_text(encoding="utf-8"))
    model = joblib.load(V1 / "model_B.joblib")
    if model.n_features_in_ != len(meta["features"]) or len(meta["features"]) != 15:
        raise ScoringError(f"model {model.n_features_in_} giriş bekliyor, features.json {len(meta['features'])} özellik")
    return dict(model=model, features=meta["features"],
                thresholds=json.loads((V1 / "thresholds.json").read_text(encoding="utf-8")),
                clean_ref=pl.read_parquet(V1 / "clean_reference.parquet"),
                ref=json.loads((REF / "reference.json").read_text(encoding="utf-8")),
                kill_oof=pl.read_parquet(REF / "kill_oof_clean.parquet"),
                clean_kills=pl.read_parquet(REF / "clean_kill_features.parquet"),
                clean_players=pl.read_parquet(REF / "clean_player_features.parquet"))


def check_phase1(work: Path, feats: list[str]) -> dict:
    """Faz 1 çıktısını doğrular. UYGUN DEĞİL → skor üretilmez. Sessiz düzeltme yapılmaz."""
    for f in ["qc.json", "metadata.json", "features.parquet", "kills.parquet"]:
        if not (work / f).exists():
            raise ScoringError(f"Faz 1 çıktısı eksik: {f}")
    qc = json.loads((work / "qc.json").read_text(encoding="utf-8"))
    cols = pl.read_parquet_schema(work / "features.parquet")
    missing = [c for c in feats if c not in cols]
    if missing:
        raise ScoringError(f"features.parquet'te eksik özellik: {missing}")
    if list(cols)[len(ID_COLS):] != feats or list(cols)[:len(ID_COLS)] != ID_COLS:
        raise ScoringError("features.parquet sütun sırası models/v1/features.json ile birebir aynı değil")
    if qc.get("feature_order") != feats:
        raise ScoringError("qc.json feature_order models/v1/features.json ile aynı değil")
    return qc


def percentile_below(ref: np.ndarray, v: float) -> float:
    """ref değerlerinin kaçının v'den KÜÇÜK olduğu (0–1)."""
    ref = ref[~np.isnan(ref)]
    return float((ref < v).mean()) if len(ref) else float("nan")


def explain(values: dict, refs: dict, dirs: dict, group_text: str, k: int = 3) -> tuple[list[dict], str]:
    """En belirleyici k özellik: 'hileci yönünde' yüzdeliği en yüksek olanlar. Deterministik (eşitlikte ad sırası)."""
    items = []
    for f, v in values.items():
        if v is None or (isinstance(v, float) and np.isnan(v)) or dirs.get(f, 0) == 0 or f not in refs:
            continue
        ref = refs[f]
        ref = ref[~np.isnan(ref)]
        if not len(ref):
            continue
        below, above = float((ref < v).mean()), float((ref > v).mean())
        notable = below if dirs[f] > 0 else above        # +1: yüksek değer dikkat çekici → v referansın çoğundan büyük
        items.append(dict(feature=f, value=float(v), notable_pct=round(notable * 100, 1), direction=dirs[f]))
    items.sort(key=lambda d: (-d["notable_pct"], d["feature"]))
    top = items[:k]
    texts = []
    for d in top:
        name, unit, fmt, low, high = FEATURE_TEXT[d["feature"]]
        word = high if d["direction"] > 0 else low
        val = fmt.format(d["value"]) + (f" {unit}" if unit else "")
        typical = " (tipik aralıkta)" if d["notable_pct"] < TYPICAL_BELOW else ""
        texts.append(f"{name} {val} — {group_text} {pct_from(int(round(d['notable_pct'])))} {word}{typical}")
        d["text"] = texts[-1]
    return top, "; ".join(texts)


def is_knife(weapon: str | None) -> bool:
    return bool(weapon) and (weapon.startswith("knife") or weapon.startswith("bayonet"))


# ---------------------------------------------------------------- skorlama
def kill_scores(feats_df: pl.DataFrame, fr: dict) -> np.ndarray:
    X = feats_df.select(fr["features"]).to_numpy().astype(float)     # eğitimle aynı: to_numpy().astype(float)
    return fr["model"].predict_proba(X)[:, 1]


def player_table(feats_df: pl.DataFrame, scores: np.ndarray, fr: dict) -> pl.DataFrame:
    """Oyuncu skoru baseline.aggregate ("mean", MIN_KILLS) ile AYNEN; < 3 öldürme 'yetersiz veri'."""
    df = feats_df.with_columns(y=pl.lit(0, pl.Int8))
    agg = aggregate(df, scores, "mean", MIN_KILLS).select("player", pl.col("score").alias("player_raw_score"))
    counts = df.group_by("player", maintain_order=True).agg(analyzed_kills=pl.len())
    styles = player_styles(df).rename({"stil": "style"})
    t = counts.join(styles, on="player", how="left").join(agg, on="player", how="left").sort("player")

    th, ref = fr["thresholds"], fr["clean_ref"]
    g0 = th["G0"]
    rows = []
    for r in t.iter_rows(named=True):
        st = r["style"]
        cs = ref.filter(pl.col("stil") == st)["score"].to_numpy()
        fallback = len(cs) < MIN_CLEAN
        if fallback:
            cs = ref["score"].to_numpy()
        p90 = float(np.quantile(cs, ORTA_Q))
        g1 = th["G1"][st]
        s = r["player_raw_score"]
        if s is None:
            prio, pct, g0r, g1r = "yetersiz veri", None, None, None
        else:
            pct = round(percentile_below(cs, s) * 100, 1)
            g0r, g1r = bool(s > g0), bool(s > g1)
            prio = ("yüksek inceleme önceliği" if s > g1 else "orta inceleme önceliği" if s > p90
                    else "düşük inceleme önceliği")
        rows.append(dict(player=r["player"], analyzed_kills=r["analyzed_kills"], style=st, style_en=STYLE_EN[st],
                         player_raw_score=s, style_percentile=pct, review_priority=prio,
                         g1_above_style_threshold=g1r, g1_threshold=g1, orta_threshold_p90=p90,
                         g0_above_threshold=g0r, g0_threshold=g0, style_reference_fallback=fallback,
                         style_reference_n=len(cs), low_confidence_style=(st == "sniper ağırlıklı")))
    return pl.DataFrame(rows)


def visibility_context(work: Path) -> pl.DataFrame | None:
    f = work / "_stages" / "visibility_windows.parquet"
    if not f.exists():
        return None
    v = pl.read_parquet(f)
    if v.height == 0:
        return None
    vv = v.filter(pl.col("is_victim"))
    last = vv.sort("tick").group_by("player", "kill_tick").agg(victim_visible_at_kill=pl.col("visible").last())
    share = vv.group_by("player", "kill_tick").agg(victim_visible_share=pl.col("visible").cast(pl.Float64).mean())
    return last.join(share, on=["player", "kill_tick"]).with_columns(
        attacker="Player_" + pl.col("player").str.split("_").list.last()).drop("player").rename({"kill_tick": "tick"})


def event_table(feats_df: pl.DataFrame, scores: np.ndarray, kills: pl.DataFrame, players: pl.DataFrame,
                vis: pl.DataFrame | None, fr: dict) -> pl.DataFrame:
    ref = fr["ref"]
    feats = fr["features"]
    e = feats_df.with_columns(event_score=pl.Series(scores)).join(
        players.select("player", "style"), on="player", how="left", maintain_order="left")
    # wallbang zaten 15 özellikten biri (penetrated > 0) → bağlamdan tekrar alınmaz
    ctx_cols = ["kill_id", "round", "victim", "headshot", "penetrated", "thrusmoke", "pre_match",
                "attacker_X", "attacker_Y", "attacker_Z", "attacker_yaw", "attacker_pitch",
                "victim_X", "victim_Y", "victim_Z", "victim_yaw", "victim_pitch"]
    ctx_cols += [c for c in ("noscope", "distance") if c in kills.columns]
    e = e.join(kills.select(ctx_cols), on="kill_id", how="left", maintain_order="left")
    if vis is not None:
        e = e.join(vis, left_on=["player", "tick"], right_on=["attacker", "tick"], how="left", maintain_order="left")
    else:
        e = e.with_columns(victim_visible_at_kill=pl.lit(None, pl.Boolean), victim_visible_share=pl.lit(None, pl.Float64))

    oof = fr["kill_oof"]
    ck = fr["clean_kills"]
    rows = []
    for r in e.iter_rows(named=True):
        st = r["style"]
        thr = ref["event_threshold"][st]
        oof_s = oof.filter(pl.col("stil") == st)["score"].to_numpy()
        knife = is_knife(r["weapon"])
        if knife:
            top, text, basis = [], ("bıçak öldürmesi (v1'de 'otomatik' sınıfına düşer, bilinen hata) — özellik "
                                    "karşılaştırması yapılmadı"), "yok (bıçak)"
        else:
            wc = r["weapon_class"]
            sub = ck.filter((pl.col("stil") == st) & (pl.col("weapon_class") == wc))
            if sub.height >= MIN_REF_KILLS:
                basis = f"temiz {st} oyuncuların {WEAPON_TR[wc]} öldürmelerinin"
            else:
                sub = ck.filter(pl.col("stil") == st)
                basis = f"temiz {st} oyuncuların öldürmelerinin"
            refs = {f: sub[f].to_numpy().astype(float) for f in ref["explain_features"]}
            top, text = explain({f: r[f] for f in ref["explain_features"]}, refs, ref["kill_direction"][wc], basis)
        rows.append(dict(kill_id=r["kill_id"], event_style_percentile=round(percentile_below(oof_s, r["event_score"]) * 100, 1),
                         review_point=bool(r["event_score"] > thr), event_threshold=thr, is_knife=knife,
                         explanation=text, explanation_basis=basis,
                         top_features=json.dumps(top, ensure_ascii=False)))
    ex = pl.DataFrame(rows)
    out = e.join(ex, on="kill_id", how="left", maintain_order="left").rename(
        {"player": "attacker", "tick": "kill_tick"}).with_columns(
        window_start_tick=pl.col("kill_tick") - 128, window_end_tick=pl.col("kill_tick"))
    order = ["kill_id", "round", "kill_tick", "attacker", "victim", "weapon", "weapon_class", "is_knife", "style",
             "event_score", "event_style_percentile", "event_threshold", "review_point", "headshot",
             "penetrated", "thrusmoke", "victim_visible_at_kill", "victim_visible_share",
             "attacker_X", "attacker_Y", "attacker_Z", "attacker_yaw", "attacker_pitch",
             "victim_X", "victim_Y", "victim_Z", "victim_yaw", "victim_pitch",
             "window_start_tick", "window_end_tick", *feats, "explanation", "explanation_basis", "top_features"]
    order += [c for c in ("noscope", "distance") if c in out.columns]
    return out.select(order).sort("kill_id")


def player_explanations(feats_df: pl.DataFrame, players: pl.DataFrame, fr: dict) -> pl.DataFrame:
    ref = fr["ref"]
    cp = fr["clean_players"]
    means = feats_df.group_by("player", maintain_order=True).agg(*[pl.col(f).mean() for f in ref["explain_features"]])
    rows = []
    for r in players.iter_rows(named=True):
        if r["player_raw_score"] is None:
            rows.append(dict(player=r["player"], player_explanation="yetersiz veri (< 3 öldürme)", player_top_features="[]"))
            continue
        st = r["style"]
        m = means.filter(pl.col("player") == r["player"]).row(0, named=True)
        sub = cp.filter(pl.col("stil") == st)
        refs = {f: sub[f].to_numpy().astype(float) for f in ref["explain_features"]}
        top, text = explain({f: m[f] for f in ref["explain_features"]}, refs, ref["player_direction"][st],
                            f"temiz {st} oyuncuların oyuncu ortalamalarının")
        rows.append(dict(player=r["player"], player_explanation=text,
                         player_top_features=json.dumps(top, ensure_ascii=False)))
    return pl.DataFrame(rows)


def map_events_table(events: pl.DataFrame, map_name: str) -> pl.DataFrame:
    """2D harita (Faz 3): her inceleme noktası + her oyuncunun en yüksek skorlu MAP_TOP öldürmesi."""
    ranked = events.with_columns(rank_in_player=pl.col("event_score").rank("ordinal", descending=True).over("attacker"))
    sel = ranked.filter(pl.col("review_point") | (pl.col("rank_in_player") <= MAP_TOP)).with_columns(
        reason=pl.when(pl.col("review_point")).then(pl.lit("inceleme noktası")).otherwise(pl.lit("oyuncunun en yüksek skorlu öldürmesi")),
        event_type=pl.lit("kill"), map=pl.lit(map_name))
    return sel.select("map", "round", pl.col("kill_tick").alias("tick"), "kill_id", "event_type", "reason",
                      "attacker", "victim", "weapon", "is_knife", "event_score", "event_style_percentile", "review_point",
                      "rank_in_player", "attacker_X", "attacker_Y", "attacker_Z", "attacker_yaw", "attacker_pitch",
                      "victim_X", "victim_Y", "victim_Z", "headshot", (pl.col("wallbang") > 0).alias("wallbang"), "thrusmoke",
                      "victim_visible_at_kill", "victim_visible_share", "reaction_ms", "prefov_angle",
                      "explanation", "window_start_tick", "window_end_tick").sort("tick", "kill_id")


# ---------------------------------------------------------------- ana akış
def build_tables(feats_df: pl.DataFrame, scores: np.ndarray, kills: pl.DataFrame, vis: pl.DataFrame | None,
                 map_name: str, fr: dict) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Öldürme skorlarından oyuncu / olay / harita tabloları. Skor kaynağı: Faz 2'de dondurulmuş model; CS2CD eğitim
    maçlarının gösteriminde değerlendirme OOF skorları — tanımlar her iki yolda aynı."""
    players = player_table(feats_df, scores, fr)
    events = event_table(feats_df, scores, kills, players, vis, fr)
    pexp = player_explanations(feats_df, players, fr)
    counts = events.group_by("attacker").agg(review_points=pl.col("review_point").sum().cast(pl.Int64))
    tops = (events.sort("event_score", descending=True).group_by("attacker", maintain_order=True)
                  .agg(top_event_ids=pl.col("kill_id").head(TOP_EVENTS), top_event_scores=pl.col("event_score").head(TOP_EVENTS)))
    players = (players.join(counts, left_on="player", right_on="attacker", how="left")
                      .join(tops, left_on="player", right_on="attacker", how="left")
                      .join(pexp, on="player", how="left").sort("player"))
    return players, events, map_events_table(events, map_name)


def write_not_applicable(work: Path, qc: dict, ident: dict, seconds: float):
    res = dict(status="skor üretilmedi", reason="Faz 1 kalite kontrolü: Model v1 için UYGUN DEĞİL",
               model_v1_reasons=qc.get("model_v1_reasons"), disclaimer=DISCLAIMER, scoring_version=SCORING_VERSION,
               model_identity=ident, demo_id=qc.get("demo_id"), seconds=seconds)
    (work / "model_result.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    (work / "model_report.md").write_text(
        f"# Model sonucu: {qc.get('demo_id')}\n\n**Skor üretilmedi.** Faz 1 kalite kontrolüne göre Model v1 bu demoya "
        f"uygulanamaz: {'; '.join(qc.get('model_v1_reasons') or [])}\n\n_{DISCLAIMER}_\n", encoding="utf-8")
    for f in ["player_scores.parquet", "event_scores.parquet", "map_events.parquet"]:
        (work / f).unlink(missing_ok=True)


def score(work: Path) -> dict:
    t0 = time.time()
    work = Path(work)
    fr = load_frozen()
    ident = model_identity()
    qc = check_phase1(work, fr["features"])
    before = {f: sha256(work / f) for f in PHASE1_FILES if (work / f).exists()}
    if not qc.get("model_v1_applicable"):
        write_not_applicable(work, qc, ident, round(time.time() - t0, 2))
        return dict(status="skor üretilmedi")

    meta = json.loads((work / "metadata.json").read_text(encoding="utf-8"))
    feats_df = pl.read_parquet(work / "features.parquet")
    kills = pl.read_parquet(work / "kills.parquet")
    if feats_df.height == 0:
        raise ScoringError("features.parquet boş")
    scores = kill_scores(feats_df, fr)
    vis = visibility_context(work)
    players, events, mev = build_tables(feats_df, scores, kills, vis, meta["map"], fr)

    players.write_parquet(work / "player_scores.parquet")
    events.write_parquet(work / "event_scores.parquet")
    mev.write_parquet(work / "map_events.parquet")
    after = {f: sha256(work / f) for f in before}
    if after != before:
        raise ScoringError("Faz 1 dosyaları değişti (olmamalı)")

    seconds = round(time.time() - t0, 2)
    prio_counts = players.group_by("review_priority").len().sort("review_priority").rows()
    res = dict(
        status="skor üretildi", disclaimer=DISCLAIMER, event_note=EVENT_NOTE, scoring_version=SCORING_VERSION,
        demo_id=meta["demo_id"], map=meta["map"], source=meta.get("source"), rounds=meta.get("rounds"),
        model_identity=ident, feature_order=fr["features"],
        decision_roles=dict(default="G1 mantığı: stil içi yüzdelik + inceleme önceliği",
                            comparison="G0: tek global eşik, yalnız karşılaştırma"),
        thresholds=dict(G0=fr["thresholds"]["G0"], G1=fr["thresholds"]["G1"],
                        event=fr["ref"]["event_threshold"], comparison="skor > eşik (değerlendirme koduyla aynı)"),
        priority_definition=dict(yuksek="skor > G1 stil eşiği (stil içi temizlerin %95'lik dilimi)",
                                 orta="skor > stil içi temizlerin %90'lık dilimi", dusuk="altı",
                                 yetersiz_veri=f"< {MIN_KILLS} öldürme"),
        kills_scored=int(events.height), review_points=int(events["review_point"].sum()),
        players=players.height, priority_counts=prio_counts,
        phase1_inputs_read=["qc.json", "metadata.json", "features.parquet", "kills.parquet"]
        + (["_stages/visibility_windows.parquet"] if vis is not None else []),
        phase1_files_verified_unchanged=sorted(before),
        visibility_context_available=vis is not None, seconds=seconds,
    )
    (work / "model_result.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    dist = distribution_info(work)
    (work / "model_report.md").write_text(report_md(res, players, events, dist), encoding="utf-8")
    return res


def distribution_info(work: Path) -> dict | None:
    """demo düzeyi dağılım bilgisi → distribution_check.json. YALNIZ GÖSTERİM: skorlar/öncelik bundan
    hesaplanmaz (skor tabloları bu çağrıdan önce yazıldı). Dondurulmuş referans yoksa None."""
    import distribution_check as DC
    if not DC.REF_FILE.exists():
        return None
    r = DC.check_analysis(work)
    (work / "distribution_check.json").write_text(json.dumps(r, ensure_ascii=False, indent=1), encoding="utf-8")
    return r


def report_md(res: dict, players: pl.DataFrame, events: pl.DataFrame, dist: dict | None = None) -> str:
    f3 = lambda x: "—" if x is None else f"{x:.3f}"   # noqa: E731
    L = [f"# Model sonucu (teknik çıktı): {res['demo_id']}", "",
         f"> **{DISCLAIMER}** Güncel demolarda kalibrasyon doğrulanmadı. Bu rapor kimseyi 'hileci' olarak "
         "nitelendirmez; yalnızca inceleme önceliği verir. Oyuncular anonimdir (Player_N).",
         f"> {EVENT_NOTE}", "",
         f"- Harita: {res['map']} | round: {res['rounds']} | skorlanan öldürme: {res['kills_scored']} | "
         f"inceleme noktası: {res['review_points']}",
         f"- Model: `models/v1/model_B.joblib` (sha256 {res['model_identity']['models/v1/model_B.joblib'][:16]}…), "
         f"sürüm {res['scoring_version']}",
         "- Varsayılan çıktı: **stil içi yüzdelik + inceleme önceliği (G1 mantığı)**. G0 (tek global eşik "
         f"{res['thresholds']['G0']:.3f}) yalnız karşılaştırma için.",
         "- İnceleme önceliği: yüksek = skor > G1 stil eşiği; orta = skor > stil içi temizlerin %90'lık dilimi; "
         f"düşük = altı; < {MIN_KILLS} öldürme = yetersiz veri.", ""]
    if dist is not None:
        import distribution_check as DC
        L += [DC.markdown(dist)]
    L += ["## Oyuncular", "",
         "| oyuncu | stil | öldürme | ham skor | stil içi yüzdelik | inceleme önceliği | G0 (karşılaştırma) | inceleme noktası |",
         "|---|---|---|---|---|---|---|---|"]
    for r in players.iter_rows(named=True):
        g0 = "—" if r["g0_above_threshold"] is None else ("eşik üstü" if r["g0_above_threshold"] else "eşik altı")
        pct = "—" if r["style_percentile"] is None else f"temizlerin {pct_from(int(round(r['style_percentile'])))} yüksek"
        L.append(f"| {r['player']} | {r['style']}{' (düşük güven)' if r['low_confidence_style'] else ''} | "
                 f"{r['analyzed_kills']} | {f3(r['player_raw_score'])} | {pct} | {r['review_priority']} | {g0} | "
                 f"{r['review_points'] or 0} |")
    L += ["", "Sniper ağırlıklı stilde ayırt etme gücü düşük (stil içi AUC ~0.75) → 'düşük güven'.", "",
          "## Oyuncu açıklamaları (en belirleyici 3 özellik)", ""]
    for r in players.iter_rows(named=True):
        L.append(f"- **{r['player']}**: {r['player_explanation']}")
    L += ["", f"## Oyuncu başına en yüksek skorlu {TOP_EVENTS} öldürme", "",
          "| oyuncu | round | tick | silah | olay skoru | stil içi olay yüzdeliği | inceleme noktası | duvar | duman | açıklama |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for p in players["player"].to_list():
        ev = events.filter(pl.col("attacker") == p).sort("event_score", descending=True).head(TOP_EVENTS)
        for e in ev.iter_rows(named=True):
            L.append(f"| {p} | {e['round']} | {e['kill_tick']} | {e['weapon']} | {e['event_score']:.3f} | "
                     f"{e['event_style_percentile']:.0f} | {'evet' if e['review_point'] else 'hayır'} | "
                     f"{'evet' if e['wallbang'] else 'hayır'} | {'evet' if e['thrusmoke'] else 'hayır'} | {e['explanation']} |")
    L += ["", f"_{DISCLAIMER}_", ""]
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Faz 2: Model B v1 skorları (kalibre edilmiş hile olasılığı DEĞİL)")
    ap.add_argument("target", help=".dem dosyası veya Faz 1 çıktı klasörü (reports/analysis/<demo_id>)")
    ap.add_argument("--source", default="other", choices=["faceit", "hltv", "valve", "other"])
    a = ap.parse_args(argv)
    target = Path(a.target)
    try:
        if target.is_dir():
            work = target
        else:
            import analyze_demo as AD
            AD.validate_file(target)
            work = AD.OUT_ROOT / AD.sha12(target)
            if not ((work / "qc.json").exists() and (work / "features.parquet").exists()):
                print("Faz 1 çıktısı yok → Faz 1 çalışıyor (görünürlük birkaç dakika sürebilir)...")
                AD.analyze(target, source=a.source)
            else:
                print(f"Faz 1 çıktısı mevcut ({work}); yeniden hesaplanmıyor.")
        res = score(work)
    except Exception as e:
        if type(e).__name__ in ("ScoringError", "AnalysisError"):
            print(f"HATA: {e}", file=sys.stderr)
            return 2
        raise
    print(f"Çıktı: {work}")
    if res["status"] != "skor üretildi":
        print("Skor üretilmedi: Faz 1 kalite kontrolü 'UYGUN DEĞİL'.")
        return 0
    print(f"{DISCLAIMER}")
    print(f"Skorlanan öldürme {res['kills_scored']} | inceleme noktası {res['review_points']} | "
          f"inceleme önceliği: {res['priority_counts']} | Faz 2 süresi {res['seconds']} sn")
    return 0


if __name__ == "__main__":
    sys.exit(main())
