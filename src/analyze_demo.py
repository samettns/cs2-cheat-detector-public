"""Faz 1: tek komutla .dem → Model B v1'in 15 özelliği + kalite kontrol + bağlam verisi.

Kullanım:
    .venv\\Scripts\\python.exe src\\analyze_demo.py <demo.dem> [--source faceit|hltv|valve|other] [--resume]
    .venv\\Scripts\\python.exe src\\analyze_demo.py --cs2cd no_cheater_present/108      (gerileme: CS2CD maçı)

Çıktı: reports/analysis/<demo_id>/  (git dışı; demo_id = dosya içeriğinin SHA-256'sının ilk 12 hanesi)
    metadata.json        demo/harita/tick hızı/round/oyuncu bilgisi, model v1 uygunluğu, çalışma süresi ve bellek
    features.parquet     öldürme başına 15 özellik, sütun sırası models/v1/features.json ile AYNI (Faz 2 girdisi)
    kills.parquet        her ölüm için bağlam (round, tick, saldırgan/kurban Player_N, silah, konum, bakış ...)
    kill_windows.parquet modele giren her öldürmede saldırgan + kurbanın son 2 sn konum/bakış izi (2 tick'te bir)
    map_context.parquet  bütün oyuncuların konumu, 8 tick'te bir (8 Hz) — 2D harita/zaman çizelgesi için
    utility.parquet      duman ve molotof olayları (konum, tick)
    qc.json / qc.md      kalite kontrol raporu
    _stages/             ara çıktılar (hata olursa kaybolmaz; --resume ile yeniden kullanılır)

Bu betik HİÇBİR oyuncuyu sınıflandırmaz ve model skoru üretmez. Model v1, özellik tanımları ve eşikler
değiştirilmez: nişan/görünürlük hesapları mevcut fonksiyonlarla (features.match_features,
visibility.match_windows, wallhack_features.compute) aynen yapılır.
Sessiz geri dönüş yoktur: kritik bir sorun AnalysisError ile durdurur; uyarılar qc raporuna yazılır.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import polars as pl
import psutil

sys.path.insert(0, str(Path(__file__).parent))
import demo_adapter as A          # noqa: E402
import wallhack_features as WF    # noqa: E402
from aim import angle_diff        # noqa: E402
from baseline import AIM_FEATURES, VISIBILITY_FEATURES, WEAPON_FEATURES  # noqa: E402
from features import NON_AIM_WEAPONS, WEAPON_CLASS, match_features       # noqa: E402
from load import RAW_DIR, event_df, load_events                          # noqa: E402
from visibility import MAPS_DIR, WINDOW, match_windows, validate          # noqa: E402

PIPELINE_VERSION = "faz1-v1"
OUT_ROOT = Path("reports/analysis")
CACHE_ROOT = Path("data/processed/demo_cache")     # dönüştürülmüş demo (tam tick tablosu, ~10 MB) + kimlik eşlemesi
MODEL_FEATURES_FILE = Path("models/v1/features.json")

EXPECTED_TICKRATE = 64.0
TICKRATE_TOL = 0.5              # 64 ± 0.5 dışında dur (v1 pencereleri 128 tick = 2 sn varsayar)
CONTEXT_STEP = 8                # map_context örnek aralığı (tick) → 8 Hz
KILL_WINDOW_STEP = 2            # kill_windows örnek aralığı (görünürlük penceresiyle aynı)
BOUNDS_MARGIN = 64.0            # harita sınırı kontrolünde pay (birim)
REQUIRED_TICK_COLUMNS = ["tick", "steamid", "X", "Y", "Z", "yaw", "pitch", "is_alive", "team_num",
                         "duck_amount", "last_place_name", "approximate_spotted_by", "game_time"]
CONTEXT_COLUMNS = ["total_rounds_played", "kills_total", "deaths_total", "is_freeze_period"]   # yoksa uyarı (model etkilenmez)
KNIFE_PREFIXES = ("knife", "bayonet")


class AnalysisError(Exception):
    """Analizi durduran, açıkça raporlanan hata (sessiz geri dönüş yok)."""


# ---------------------------------------------------------------- yardımcılar
def model_features() -> list[str]:
    """Model v1'in özellik sırası. Eğitim kodundaki listelerle aynı olmalı; değilse durur."""
    feats = json.loads(MODEL_FEATURES_FILE.read_text(encoding="utf-8"))["features"]
    if feats != AIM_FEATURES + WEAPON_FEATURES + VISIBILITY_FEATURES:
        raise AnalysisError("models/v1/features.json ile baseline özellik listeleri farklı")
    return feats


def sha12(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


def git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip() or None
    except OSError:
        return None


@lru_cache(maxsize=16)
def map_bounds(map_name: str) -> tuple[list[float], list[float]] | None:
    """Harita geometrisinin (.tri: float32 üçgen köşeleri) min/max koordinatları. BVH kurmaz (RAM dostu)."""
    f = Path(MAPS_DIR) / f"{map_name}.tri"
    if not f.exists():
        return None
    v = np.fromfile(f, dtype=np.float32).reshape(-1, 3)
    return v.min(0).tolist(), v.max(0).tolist()


def peak_ram_mb() -> float:
    mi = psutil.Process().memory_info()
    return round(getattr(mi, "peak_wset", mi.rss) / 2**20, 0)   # Windows: işlem ömrü boyunca tepe değer


def match_start_tick(ticks: pl.DataFrame) -> int | None:
    """Maç başlangıcı = skor tablosunun (oyuncuların kills_total + deaths_total toplamı) sıfır olduğu SON tick.
    Öncesi ısınma / bıçak roundu: CS2CD eğitim maçlarında bu dönemde öldürme yok (MODEL_CARD.md)."""
    if not all(c in ticks.columns for c in ("kills_total", "deaths_total")):
        return None
    per = ticks.group_by("tick").agg(s=(pl.col("kills_total").fill_null(0) + pl.col("deaths_total").fill_null(0)).sum())
    zero = per.filter(pl.col("s") == 0)
    return int(zero["tick"].max()) if zero.height else int(per["tick"].min())


def round_timeline(ticks: pl.DataFrame, start: int | None) -> pl.DataFrame | None:
    """Tick başına round numarası ve aşama: 'pre_match' / 'freeze' / 'live' / 'post'.

    total_rounds_played round'u bitiren öldürmeyle AYNI tick'te artar → "sayaç + 1" round'u bitiren öldürmeyi ve round
    sonrasını bir sonraki round'a yazıyordu. Doğrusu: round, donma süresinin bitişiyle (is_freeze_period
    True→False) başlar ve bir sonraki donma süresine kadar sürer:
      donma tick'i          → round = sayaç + 1 (başlayacak round), aşama 'freeze'
      donma dışı tick       → round = (son donma bitişindeki sayaç) + 1; sayaç o andan beri arttıysa aşama 'post'
      maç öncesi            → round 0 (maçın ilk round'unun donma başlangıcından önce; ısınma / bıçak roundu)
    Model girdisini belirleyen pre_match tanımı (öldürme tick'i <= match_start_tick) DEĞİŞMEZ."""
    if not {"total_rounds_played", "is_freeze_period"} <= set(ticks.columns):
        return None
    g = (ticks.group_by("tick").agg(trp=pl.col("total_rounds_played").max(),
                                    frz=pl.col("is_freeze_period").fill_null(False).any()).sort("tick"))
    prev = pl.col("frz").shift(1).fill_null(False)
    g = g.with_columns(live_start=prev & ~pl.col("frz"), freeze_start=~prev & pl.col("frz"))
    g = g.with_columns(ls_tick=pl.when(pl.col("live_start")).then(pl.col("tick")).forward_fill(),
                       ls_trp=pl.when(pl.col("live_start")).then(pl.col("trp")).forward_fill())
    g = g.with_columns(
        round=pl.when(pl.col("frz") | pl.col("ls_tick").is_null()).then(pl.col("trp") + 1).otherwise(pl.col("ls_trp") + 1),
        phase=pl.when(pl.col("frz")).then(pl.lit("freeze"))
                .when(pl.col("ls_tick").is_not_null() & (pl.col("trp") > pl.col("ls_trp"))).then(pl.lit("post"))
                .otherwise(pl.lit("live")))
    # Maç öncesi sınırı: ilk sayılan ölümü içeren round'un donma başlangıcı (yoksa match_start_tick)
    pre_end = start
    if start is not None:
        ls = g.filter(pl.col("live_start") & (pl.col("tick") <= start + 1))
        if ls.height:
            l1 = ls["tick"].max()
            fs = g.filter(pl.col("freeze_start") & (pl.col("tick") <= l1))
            pre_end = (fs["tick"].max() - 1) if fs.height else start
        g = g.with_columns(round=pl.when(pl.col("tick") <= pre_end).then(0).otherwise(pl.col("round")),
                           phase=pl.when(pl.col("tick") <= pre_end).then(pl.lit("pre_match")).otherwise(pl.col("phase")))
    return g.select("tick", pl.col("round").cast(pl.Int32), "phase")


# ---------------------------------------------------------------- aşamalar
def validate_file(path: Path) -> None:
    if not path.exists():
        raise AnalysisError(f"dosya yok: {path}")
    if path.suffix.lower() != ".dem":
        raise AnalysisError(f"'.dem' uzantılı değil: {path.name}")
    with open(path, "rb") as f:
        magic = f.read(8)
    if not magic.startswith(b"PBDEMS2"):
        raise AnalysisError(f"CS2 demo başlığı (PBDEMS2) yok: {path.name} (bozuk/sıkıştırılmış veya CS:GO demosu)")


def convert_stage(path: Path, source: str, did: str, cache: Path, resume: bool) -> tuple[Path, dict]:
    """Adaptör: .dem → CS2CD şeması (anonim Player_N), önbellek klasörüne."""
    out = cache / did
    match = out / did
    meta_f = out / "adapter_meta.json"
    if resume and meta_f.exists() and match.with_suffix(".parquet").exists():
        return match, json.loads(meta_f.read_text(encoding="utf-8"))
    try:
        meta = A.convert(source, path, 0, out_dir=out, match_id=did)
    except Exception as e:
        raise AnalysisError(f"demo ayrıştırılamadı ({type(e).__name__}: {str(e)[:200]})") from e
    meta_f.write_text(json.dumps(meta, default=str), encoding="utf-8")
    return match, meta


def gate_checks(match: Path, events: dict) -> tuple[dict, pl.DataFrame]:
    """Temel kalite kontrolleri. Kritik sorunlar AnalysisError; diğerleri 'warnings'."""
    schema = pl.read_parquet_schema(match.with_suffix(".parquet"))
    missing_cols = [c for c in REQUIRED_TICK_COLUMNS if c not in schema]
    if missing_cols:
        raise AnalysisError(f"zorunlu tick sütunları eksik: {missing_cols}")
    missing_ev = {e: ([] if e in events else ["(olay yok)"]) for e in A.REQUIRED_EVENTS}
    for e, fields in A.REQUIRED_EVENTS.items():
        if e in events and events[e]:
            missing_ev[e] = [f for f in fields if f not in events[e][0]]
        elif e in events:
            missing_ev[e] = ["(olay boş)"]
    missing_ev = {e: v for e, v in missing_ev.items() if v}
    if missing_ev:
        raise AnalysisError(f"zorunlu olay/alan eksik: {missing_ev}")

    cols = ["tick", "steamid", "team_num", "is_alive", "game_time"] + [c for c in CONTEXT_COLUMNS if c in schema]
    t = pl.read_parquet(match.with_suffix(".parquet"), columns=cols)
    gt = t.group_by("tick").agg(pl.col("game_time").first()).sort("tick").drop_nulls("game_time")
    if gt.height < 2 or gt["game_time"][-1] == gt["game_time"][0]:
        raise AnalysisError("tick hızı ölçülemedi (game_time yok/sabit)")
    tickrate = round((gt["tick"][-1] - gt["tick"][0]) / (gt["game_time"][-1] - gt["game_time"][0]), 2)
    if abs(tickrate - EXPECTED_TICKRATE) > TICKRATE_TOL:
        raise AnalysisError(f"tick hızı {tickrate}, beklenen {EXPECTED_TICKRATE}: v1 pencereleri (128 tick = 2 sn) "
                            "geçersiz olur → model v1 bu demoya uygulanamaz")
    tk = t["tick"].unique().sort()
    gaps = tk.diff().drop_nulls().value_counts().sort("count", descending=True).head(3).rows()

    warnings = []
    players = t.filter(pl.col("team_num").is_in([2, 3]))
    per_tick = players.group_by("tick", "team_num").agg(n=pl.col("steamid").n_unique())
    ppt = per_tick.group_by("team_num").agg(med=pl.col("n").median(), mx=pl.col("n").max()).sort("team_num")
    n_players = players["steamid"].n_unique()
    if ppt.height != 2:
        warnings.append(f"takım sayısı {ppt.height} (beklenen 2)")
    if any(m != 5 for m in ppt["med"].to_list()):
        warnings.append(f"takım başına anlık oyuncu medyanı {ppt.rows()} (beklenen 5)")
    if any(m > 5 for m in ppt["mx"].to_list()):
        warnings.append(f"takım başına anlık oyuncu en fazla {ppt['mx'].to_list()} (5'ten fazla)")
    if n_players != 10:
        warnings.append(f"maç boyunca {n_players} farklı oyuncu (beklenen 10; oyuncu değişimi/bağlantı kopması olabilir)")
    if gaps and gaps[0][0] != 1:
        warnings.append(f"ardışık tick farkı çoğunlukla {gaps[0][0]} (beklenen 1)")
    missing_ctx = [c for c in CONTEXT_COLUMNS if c not in schema]
    if missing_ctx:
        warnings.append(f"bağlam sütunları yok {missing_ctx}: round numarası / maç başlangıcı belirlenemez")
    start = match_start_tick(t)
    last = t.filter(pl.col("tick") == t["tick"].max())
    rounds = int(last["total_rounds_played"].max()) if "total_rounds_played" in t.columns else None
    return dict(tickrate=tickrate, tick_gaps=gaps, minutes=round(tk.len() / tickrate / 60, 1),
                n_players=n_players, players_per_team=ppt.rows(), match_start_tick=start, rounds=rounds,
                first_tick=int(tk[0]), last_tick=int(tk[-1]), warnings=warnings), t


def kills_table(events: dict, ticks: pl.DataFrame, start: int | None, timeline: pl.DataFrame | None = None) -> pl.DataFrame:
    """Bütün ölümler (bağlam). kill_id = olay sırası; 'occ' aynı tick + saldırgandaki sıra (çoklu öldürme)."""
    d = event_df(events, "player_death").with_row_index("kill_id")
    keep = ["kill_id", "tick", "attacker_steamid", "user_steamid", "weapon", "headshot", "penetrated", "thrusmoke"]
    keep += [c for c in ("noscope", "attackerblind", "distance", "hitgroup") if c in d.columns]
    d = d.select(keep).rename({"attacker_steamid": "attacker", "user_steamid": "victim"})
    d = d.with_columns(
        weapon_class=pl.col("weapon").replace_strict(WEAPON_CLASS, default="otomatik"),
        wallbang=(pl.col("penetrated").fill_null(0) > 0),
        aim_weapon=~pl.col("weapon").is_in(list(NON_AIM_WEAPONS)),
        knife_skin=pl.col("weapon").str.starts_with(KNIFE_PREFIXES[0]) | pl.col("weapon").str.starts_with(KNIFE_PREFIXES[1]),
        pre_match=(pl.col("tick") <= start) if start is not None else pl.lit(None, pl.Boolean),
    ).with_columns(
        knife_skin=pl.col("knife_skin") & pl.col("aim_weapon"),   # v1: NON_AIM dışı bıçak skini → 'otomatik'
        # features.match_features ile aynı filtre: saldırgan var, intihar değil, nişan silahı
        eligible=pl.col("attacker").is_not_null() & (pl.col("attacker") != pl.col("victim")) & pl.col("aim_weapon"),
    ).with_columns(occ=pl.int_range(pl.len()).over("tick", "attacker", "eligible"))
    # Round: round_timeline (donma bitişlerine göre). Maç öncesi öldürme (pre_match) → 0.
    # Öldürmenin round'u/aşaması bir ÖNCEKİ tick'teki oyun durumundan: round'u (ve maçı) bitiren öldürmenin tick'inde
    # sayaç artar ve maç sonunda donma başlar; öldürme anındaki durum bir önceki tick'tir.
    if timeline is not None:
        d = (d.with_columns(prev_tick=pl.col("tick") - 1)
              .join(timeline.select(prev_tick="tick", tl_round="round", phase="phase"), on="prev_tick", how="left",
                    maintain_order="left")
              .with_columns(round=pl.when(pl.col("pre_match").fill_null(False)).then(0).otherwise(pl.col("tl_round")),
                            phase=pl.when(pl.col("pre_match").fill_null(False)).then(pl.lit("pre_match"))
                                    .otherwise(pl.col("phase")))
              .drop("tl_round", "prev_tick"))
    elif "total_rounds_played" in ticks.columns:   # eski yedek (is_freeze_period yoksa; qc uyarır)
        rp = ticks.group_by("tick").agg(trp=pl.col("total_rounds_played").max())
        d = d.join(rp, on="tick", how="left", maintain_order="left").with_columns(
            round=pl.when(pl.col("pre_match").fill_null(False)).then(0).otherwise(pl.col("trp") + 1)).drop("trp")
    else:
        d = d.with_columns(round=pl.lit(None, pl.Int64))
    return d


def positions_at(match: Path, keys: pl.DataFrame, cols: list[str]) -> pl.DataFrame:
    """Sadece istenen (tick, steamid) satırlarını okur (bütün maçı RAM'e almaz)."""
    ticks = keys["tick"].unique().to_list()
    t = (pl.scan_parquet(match.with_suffix(".parquet")).select("tick", "steamid", *cols)
           .filter(pl.col("tick").is_in(ticks)).collect())
    return keys.join(t, on=["tick", "steamid"], how="left", maintain_order="left")


def kill_positions(match: Path, kills: pl.DataFrame) -> pl.DataFrame:
    """Öldürme tick'inde saldırgan ve kurbanın konum/bakışı + iki tarafın takımı."""
    cols = ["X", "Y", "Z", "yaw", "pitch", "team_num", "is_alive"]
    out = kills
    for role, col in (("attacker", "attacker"), ("victim", "victim")):
        k = kills.select("kill_id", "tick", pl.col(col).alias("steamid"))
        p = positions_at(match, k.select("tick", "steamid").unique(maintain_order=True), cols)
        k = k.join(p, on=["tick", "steamid"], how="left", maintain_order="left")
        out = out.join(k.select("kill_id", *[pl.col(c).alias(f"{role}_{c}") for c in cols]), on="kill_id",
                       how="left", maintain_order="left")
    # Bir önceki tick'te ikisi de hayatta mı? (adaptör tutarlılık testi için)
    prev = kills.select("kill_id", tick=pl.col("tick") - 1, a=pl.col("attacker"), v=pl.col("victim"))
    for role, col in (("attacker", "a"), ("victim", "v")):
        k = prev.select("kill_id", "tick", pl.col(col).alias("steamid"))
        p = positions_at(match, k.select("tick", "steamid").unique(maintain_order=True), ["is_alive"])
        k = k.join(p, on=["tick", "steamid"], how="left", maintain_order="left")
        out = out.join(k.select("kill_id", pl.col("is_alive").alias(f"{role}_alive_prev")), on="kill_id",
                       how="left", maintain_order="left")
    return out


def aim_stage(match: Path) -> pl.DataFrame:
    """Nişan + silah özellikleri: features.match_features AYNEN; is_sniper/is_pistol baseline.load ile aynı ifade."""
    rows = match_features(match)
    if not rows:
        raise AnalysisError("modele girebilecek öldürme yok (nişan özelliği hesaplanan öldürme 0)")
    return pl.DataFrame(rows).with_columns(
        is_sniper=(pl.col("weapon_class") == "sniper").cast(pl.Int8),
        is_pistol=(pl.col("weapon_class") == "tabanca").cast(pl.Int8),
        occ=pl.int_range(pl.len()).over("player", "tick"))


def visibility_stage(match_key: str, base: Path, aim: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Görünürlük pencereleri (ray-cast) + 5 görünürlük özelliği; mevcut fonksiyonlar aynen."""
    vis = match_windows(match_key, kills=aim.select("match", "player", "tick"), base=base)
    if vis.height == 0:
        return vis, pl.DataFrame(schema={"match": pl.Utf8, "player": pl.Utf8, "tick": pl.Int64,
                                         **{c: pl.Float64 for c in VISIBILITY_FEATURES}})
    vf = WF.compute(vis, lambda m: load_events(base / m)).rename({"kill_tick": "tick"})
    return vis, vf.select("match", "player", "tick", *VISIBILITY_FEATURES)


def assemble(aim: pl.DataFrame, vf: pl.DataFrame | None, feats: list[str]) -> pl.DataFrame:
    """15 özelliği models/v1/features.json sırasıyla birleştirir (features.attach_visibility ile aynı sol join)."""
    if vf is None:   # desteklenmeyen harita: görünürlük UYDURULMAZ, null kalır
        out = aim.with_columns([pl.lit(None, pl.Float64).alias(c) for c in VISIBILITY_FEATURES])
    else:
        out = aim.join(vf, on=["match", "player", "tick"], how="left", maintain_order="left")
    if out.height != aim.height:
        raise AnalysisError("özellik birleştirme satır sayısını değiştirdi")
    return out


def kill_windows(match: Path, kills: pl.DataFrame) -> pl.DataFrame:
    """Modele giren öldürmelerde saldırgan + kurbanın son 2 sn izi (2 tick'te bir)."""
    rows = []
    for r in kills.iter_rows(named=True):
        for t in range(r["tick"] - WINDOW, r["tick"] + 1, KILL_WINDOW_STEP):
            rows.append((r["kill_id"], t, t - r["tick"], "attacker", r["attacker"]))
            rows.append((r["kill_id"], t, t - r["tick"], "victim", r["victim"]))
    keys = pl.DataFrame(rows, schema=["kill_id", "tick", "dt", "role", "steamid"], orient="row")
    p = positions_at(match, keys.select("tick", "steamid").unique(maintain_order=True),
                     ["X", "Y", "Z", "yaw", "pitch", "is_alive"])
    return keys.join(p, on=["tick", "steamid"], how="left", maintain_order="left").rename({"steamid": "player"})


def map_context(match: Path, start: int | None, timeline: pl.DataFrame | None = None) -> pl.DataFrame:
    """2D harita için hafif zaman çizelgesi: her oyuncu, CONTEXT_STEP tick'te bir."""
    schema = pl.read_parquet_schema(match.with_suffix(".parquet"))
    cols = [c for c in ["X", "Y", "Z", "yaw", "pitch", "is_alive", "team_num", "health", "active_weapon_name",
                        "last_place_name", "total_rounds_played"] if c in schema]
    t = (pl.scan_parquet(match.with_suffix(".parquet")).select("tick", "steamid", *cols)
           .filter((pl.col("tick") % CONTEXT_STEP == 0) & pl.col("team_num").is_in([2, 3])).collect())
    if timeline is not None:   # round ve aşama (pre_match/freeze/live/post); 2D yollar yalnız 'live'
        t = t.join(timeline, on="tick", how="left", maintain_order="left").drop(
            [c for c in ["total_rounds_played"] if c in t.columns])
    elif "total_rounds_played" in t.columns:
        pre = (pl.col("tick") <= start) if start is not None else pl.lit(False)
        t = t.with_columns(round=pl.when(pre).then(0).otherwise(pl.col("total_rounds_played") + 1)).drop("total_rounds_played")
    return t.rename({"steamid": "player"}).sort("tick", "player")


def utility_table(events: dict) -> pl.DataFrame:
    parts = []
    for name in ("smokegrenade_detonate", "smokegrenade_expired", "inferno_startburn", "inferno_expire"):
        if events.get(name):
            parts.append(event_df(events, name).select("tick", "entityid", "x", "y", "z",
                                                         pl.col("user_steamid").alias("player"))
                         .with_columns(event=pl.lit(name)))
    return (pl.concat(parts, how="diagonal_relaxed").sort("tick", "event", "entityid") if parts
            else pl.DataFrame(schema={"tick": pl.Int64, "entityid": pl.Int64, "x": pl.Float64, "y": pl.Float64,
                                      "z": pl.Float64, "player": pl.Utf8, "event": pl.Utf8}))


def aim_step_p99(match: Path) -> float:
    """Canlı oyuncularda tick'ten tick'e nişan açısı değişiminin P99'u (demo_compat.tick_stats ile aynı tanım)."""
    t = (pl.read_parquet(match.with_suffix(".parquet"), columns=["tick", "steamid", "yaw", "pitch", "is_alive"])
           .filter(pl.col("is_alive").fill_null(False)).sort("steamid", "tick"))
    d = t.with_columns(dy=angle_diff(pl.col("yaw"), pl.col("yaw").shift(1)).over("steamid"),
                       dp=(pl.col("pitch") - pl.col("pitch").shift(1)).over("steamid"),
                       dt=(pl.col("tick") - pl.col("tick").shift(1)).over("steamid")).filter(pl.col("dt") == 1)
    return round(float(((d["dy"] ** 2 + d["dp"] ** 2).sqrt()).quantile(0.99)), 3)


def raycast_checks(match_key: str, base: Path, vis: pl.DataFrame, events: dict) -> dict:
    """bullet_damage varsa eski ölçüt (CS2CD ile doğrudan karşılaştırılabilir); her durumda yedek ölçüt."""
    out = {}
    if vis.height:
        d = event_df(events, "player_death").select("tick", "attacker_steamid", "penetrated", "thrusmoke")
        last = (vis.filter(pl.col("is_victim")).sort("tick").group_by("match", "player", "kill_tick").agg(pl.all().last())
                   .with_columns(sid="Player_" + pl.col("player").str.split("_").list.last()))
        j = last.join(d, left_on=["kill_tick", "sid"], right_on=["tick", "attacker_steamid"], how="left")
        clean = j.filter((pl.col("penetrated").fill_null(0) == 0) & ~pl.col("thrusmoke").fill_null(False)).drop_nulls("visible")
        out["proxy"] = dict(nonpen_n=clean.height,
                            nonpen_visible=round(clean["visible"].mean(), 3) if clean.height else None)
    if events.get("bullet_damage"):
        v = validate(match_key, base=base).filter(~pl.col("masked"))
        np_, pn = v.filter(pl.col("pen") == 0), v.filter(pl.col("pen") > 0)
        out["bullet_damage"] = dict(nonpen_n=np_.height,
                                    nonpen_visible=round(np_["any"].mean(), 3) if np_.height else None,
                                    pen_n=pn.height, pen_visible=round(pn["any"].mean(), 3) if pn.height else None)
    return out


# ---------------------------------------------------------------- ana akış
def analyze(dem: Path | None = None, cs2cd: str | None = None, source: str = "other",
            out_root: Path = OUT_ROOT, cache_root: Path = CACHE_ROOT, resume: bool = False,
            write_md: bool = True, raw_dir: Path = RAW_DIR) -> Path:
    t_start = time.time()
    timing = {}
    feats_order = model_features()

    def lap(name):
        timing[name] = round(time.time() - lap.t, 1)
        lap.t = time.time()
    lap.t = time.time()

    # 1) Dosya doğrulama + adaptör
    if cs2cd:   # gerileme modu: CS2CD formatındaki maç (adaptör atlanır)
        did = "cs2cd_" + cs2cd.replace("/", "_")
        match = Path(raw_dir) / cs2cd
        if not match.with_suffix(".parquet").exists():
            raise AnalysisError(f"CS2CD maçı yok: {cs2cd}")
        match_key, base = cs2cd, Path(raw_dir)
        adapter_meta = dict(source="cs2cd", match_id=did)
    else:
        dem = Path(dem)
        validate_file(dem)
        did = sha12(dem)
        match, adapter_meta = convert_stage(dem, source, did, cache_root, resume)
        match_key, base = f"{did}/{did}", cache_root
    lap("donusturme")
    work = out_root / did
    stages = work / "_stages"
    stages.mkdir(parents=True, exist_ok=True)

    events = load_events(match)
    map_name = events["CSstats_info"][0]["map"]

    # 2) Temel kalite kontrolleri (kritikse durur)
    gates, ticks_small = gate_checks(match, events)
    lap("kontroller")

    # 3) Ölüm bağlamı + maç başlangıcı
    start = gates["match_start_tick"]
    timeline = round_timeline(ticks_small, start)
    kills = kills_table(events, ticks_small, start, timeline)
    kills = kill_positions(match, kills)
    del ticks_small

    # 4) Nişan + silah özellikleri
    f_aim = stages / "aim.parquet"
    if resume and f_aim.exists():
        aim = pl.read_parquet(f_aim)
    else:
        aim = aim_stage(match)
        aim.write_parquet(f_aim)
    # öldürme kimliği: (tick, saldırgan, aynı tick'teki sıra)
    aim = aim.with_columns(attacker="Player_" + pl.col("player").str.split("_").list.last()).join(
        kills.filter(pl.col("eligible")).select("kill_id", "tick", "attacker", "occ", "pre_match"),
        on=["tick", "attacker", "occ"],
        how="left", maintain_order="left")
    if aim["kill_id"].null_count():
        raise AnalysisError(f"{aim['kill_id'].null_count()} öldürme ölüm olaylarıyla eşleşmedi")
    n_pre = int(aim["pre_match"].fill_null(False).sum())
    # Maç öncesi (ısınma/bıçak roundu) öldürmeler model girdisinden çıkar: CS2CD eğitiminde yoktu.
    aim_model = aim.filter(~pl.col("pre_match").fill_null(False))
    lap("nisan")

    # 5-6) Görünürlük pencereleri + görünürlük özellikleri
    tri_ok = (Path(MAPS_DIR) / f"{map_name}.tri").exists()
    f_vis, f_vf = stages / "visibility_windows.parquet", stages / "visibility_features.parquet"
    if not tri_ok:
        vis, vf = pl.DataFrame(), None
    elif resume and f_vis.exists() and f_vf.exists():
        vis, vf = pl.read_parquet(f_vis), pl.read_parquet(f_vf)
    else:
        vis, vf = visibility_stage(match_key, base, aim_model.drop("attacker", "kill_id", "occ", "pre_match"))
        vis.write_parquet(f_vis)
        vf.write_parquet(f_vf)
    lap("gorunurluk")

    # 7) 15 özellik, model sırası
    full = assemble(aim_model, vf, feats_order)
    features = full.select("kill_id", "match", pl.col("attacker").alias("player"), "tick", "weapon", "weapon_class",
                           *feats_order).sort("kill_id")
    features.write_parquet(work / "features.parquet")

    # Bağlam tabloları
    kills = kills.with_columns(in_model=pl.col("kill_id").is_in(features["kill_id"].to_list()))
    kills.drop("occ").write_parquet(work / "kills.parquet")
    kw = kill_windows(match, kills.filter(pl.col("in_model")))
    kw.write_parquet(work / "kill_windows.parquet")
    ctx = map_context(match, start, timeline)
    ctx.write_parquet(work / "map_context.parquet")
    utility_table(events).write_parquet(work / "utility.parquet")
    lap("baglam")

    # 8) Kalite kontrol
    rc = raycast_checks(match_key, base, vis, events) if tri_ok else {}
    lap("raycast_kontrol")
    nonnull = {c: round(float(features[c].is_not_null().mean()), 3) for c in feats_order}
    bounds = map_bounds(map_name)
    oob = None
    if bounds:
        lo, hi = bounds
        oob = ctx.filter(pl.col("is_alive").fill_null(False)).filter(
            (pl.col("X") < lo[0] - BOUNDS_MARGIN) | (pl.col("X") > hi[0] + BOUNDS_MARGIN)
            | (pl.col("Y") < lo[1] - BOUNDS_MARGIN) | (pl.col("Y") > hi[1] + BOUNDS_MARGIN)
            | (pl.col("Z") < lo[2] - BOUNDS_MARGIN) | (pl.col("Z") > hi[2] + BOUNDS_MARGIN)).height
    reasons = []
    if not tri_ok:
        reasons.append(f"harita {map_name} desteklenmiyor (.tri yok): 5 görünürlük özelliği hesaplanamadı (null)")
    missing_feats = [c for c in feats_order if c not in features.columns]
    if missing_feats:
        reasons.append(f"eksik özellik sütunu: {missing_feats}")
    all_null = [c for c in feats_order if nonnull[c] == 0.0]
    if all_null and tri_ok:
        reasons.append(f"tamamen boş özellik: {all_null}")
    if features.height == 0:
        reasons.append("modele girecek öldürme yok")
    warnings = list(gates["warnings"])
    if start is None:
        warnings.append("maç başlangıcı belirlenemedi: ısınma/bıçak roundu öldürmeleri ayıklanamadı")
    if oob:
        warnings.append(f"{oob} canlı oyuncu konumu harita geometrisi sınırları dışında (±{BOUNDS_MARGIN:.0f})")
    knife_model = int(features["weapon"].str.starts_with("knife").sum() + features["weapon"].str.starts_with("bayonet").sum())
    if knife_model:
        warnings.append(f"modele giren {knife_model} bıçak skini öldürmesi 'otomatik' sayıldı (v1 bilinen hatası)")

    qc = dict(
        demo_id=did, source=adapter_meta.get("source"), map=map_name, map_supported=tri_ok,
        patch_version=adapter_meta.get("patch_version"), demo_format=adapter_meta.get("demo_version_name"),
        tickrate=gates["tickrate"], tick_gaps=gates["tick_gaps"], minutes=gates["minutes"],
        rounds=gates["rounds"], match_start_tick=start, n_players=gates["n_players"],
        players_per_team=gates["players_per_team"],
        deaths_total=kills.height, kills_pre_match=int(kills["pre_match"].fill_null(False).sum()),
        kills_aim_features=aim.height, kills_pre_match_aim=n_pre, kills_model=features.height,
        players_in_model=features["player"].n_unique(),
        teamkills=int(kills.filter(pl.col("attacker").is_not_null() & (pl.col("attacker") != pl.col("victim"))
                                   & (pl.col("attacker_team_num") == pl.col("victim_team_num"))).height),
        feature_order=feats_order, feature_nonnull=nonnull,
        feature_nan_rate={c: round(1 - v, 3) for c, v in nonnull.items()},
        visibility_computable=tri_ok,
        raycast=rc,
        raycast_note=("Yedek ölçüt (öldürmeden önceki son örnek anda kurban görünür mü; delmeyen + dumansız öldürmeler) "
                      "eski bullet_damage ölçütüyle (her delmeyen isabet, isabet anında) doğrudan eşdeğer DEĞİLDİR."),
        aim_step_p99=aim_step_p99(match),
        knife_skin_kills_total=int(kills["knife_skin"].sum()),
        knife_skin_kills_model=knife_model,
        knife_note="v1 bilinen hatası: bıçak skinleri (knife_*) nişan gerektirmeyen silah listesinde yok → 'otomatik' "
                   "sınıfına düşer; CS2CD eğitiminde de aynıydı. Model v1 donuk olduğu için düzeltilmedi.",
        positions_out_of_bounds=oob,
        warnings=warnings,
        model_v1_applicable=not reasons,
        model_v1_reasons=reasons,
    )
    (work / "qc.json").write_text(json.dumps(qc, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    if write_md:
        (work / "qc.md").write_text(qc_markdown(qc), encoding="utf-8")

    rounds_tbl = []
    if "round" in ctx.columns:
        agg = [pl.col("tick").min().alias("start_tick"), pl.col("tick").max().alias("end_tick")]
        if "phase" in ctx.columns:
            live = pl.col("tick").filter(pl.col("phase") == "live")
            agg += [live.min().alias("live_start_tick"), live.max().alias("live_end_tick")]
        rounds_tbl = (ctx.filter(pl.col("round") > 0).group_by("round").agg(*agg).sort("round").rows(named=True))
    metadata = dict(
        demo_id=did, pipeline_version=PIPELINE_VERSION, git_commit=git_commit(), source=qc["source"], map=map_name,
        patch_version=qc["patch_version"], demo_format=qc["demo_format"], tickrate=qc["tickrate"],
        minutes=qc["minutes"], rounds=qc["rounds"], rounds_table=rounds_tbl, match_start_tick=start,
        n_players=qc["n_players"], kills_model=qc["kills_model"], feature_order=feats_order,
        model_v1_applicable=qc["model_v1_applicable"], model_v1_reasons=reasons,
        context_sampling=dict(map_context_step_ticks=CONTEXT_STEP, kill_window_ticks=WINDOW,
                              kill_window_step_ticks=KILL_WINDOW_STEP),
        files=sorted(p.name for p in work.iterdir() if p.is_file()) + ["metadata.json"],
        run=dict(seconds_total=round(time.time() - t_start, 1), seconds_by_stage=timing, peak_ram_mb=peak_ram_mb()),
    )
    metadata["files"] = sorted(set(metadata["files"]))
    (work / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    return work


def qc_markdown(q: dict) -> str:
    ok = lambda b: "✅" if b else "❌"   # noqa: E731
    rc = q["raycast"]
    L = [f"# Kalite kontrol: {q['demo_id']}", "",
         f"**Model v1 için: {'UYGUN' if q['model_v1_applicable'] else 'UYGUN DEĞİL'}** "
         + ("" if q["model_v1_applicable"] else "— " + "; ".join(q["model_v1_reasons"])),
         "", "_Bu rapor oyuncu sınıflandırması ve model skoru içermez. Oyuncular anonimdir (Player_N)._", "",
         "## Demo",
         f"- Kaynak: {q['source']} | harita: {q['map']} (destekli {ok(q['map_supported'])}) | patch {q['patch_version']} | "
         f"format {q['demo_format']}",
         f"- Tick hızı: **{q['tickrate']}** (beklenen 64) | ardışık tick farkları {q['tick_gaps']} | süre {q['minutes']} dk",
         f"- Round: {q['rounds']} | maç başlangıç tick'i: {q['match_start_tick']} | oyuncu: {q['n_players']} | "
         f"takım başına anlık (takım, medyan, max): {q['players_per_team']}",
         "", "## Öldürmeler",
         f"- Bütün ölümler: {q['deaths_total']} (maç öncesi: {q['kills_pre_match']}, takım öldürme: {q['teamkills']})",
         f"- Nişan özelliği hesaplanan: {q['kills_aim_features']} → maç öncesi çıkarılan: {q['kills_pre_match_aim']} → "
         f"**modele girecek: {q['kills_model']}** ({q['players_in_model']} oyuncu)",
         "", "## 15 özellik (models/v1/features.json sırası)", "", "| # | özellik | dolu | NaN |", "|---|---|---|---|"]
    L += [f"| {i + 1} | {c} | {q['feature_nonnull'][c]:.3f} | {q['feature_nan_rate'][c]:.3f} |"
          for i, c in enumerate(q["feature_order"])]
    L += ["", "NaN'lar doldurulmaz (model native NaN ile eğitildi). Pre-FOV/tepki/görünmezken açı NaN'ları "
          "tanım gereği (ör. görünürlük geçişi yok) oluşur; CS2CD'de de benzer oranlarda.",
          "", "## Görünürlük ve ray-cast",
          f"- Görünürlük hesaplanabilir: {ok(q['visibility_computable'])}"]
    if "bullet_damage" in rc:
        b = rc["bullet_damage"]
        L.append(f"- **Eski ölçüt (`bullet_damage`)**: delmeyen isabetlerde görünür {b['nonpen_visible']} (n={b['nonpen_n']}); "
                 f"delenlerde {b['pen_visible']} (n={b['pen_n']}). CS2CD 9 haritada delmeyen %95–99.")
    else:
        L.append("- `bullet_damage` olayı yok → eski ölçüt hesaplanamaz.")
    if "proxy" in rc:
        L.append(f"- Yedek ölçüt: {rc['proxy']['nonpen_visible']} (n={rc['proxy']['nonpen_n']})")
    L += [f"- ⚠️ {q['raycast_note']}",
          "", "## Diğer",
          f"- Nişan değişimi P99 (tick'ten tick'e açı): {q['aim_step_p99']} (CS2CD maç başı tipik aralık ~4.8–8.9)",
          f"- Harita sınırı dışındaki canlı konum (8 Hz örnek): {q['positions_out_of_bounds']}",
          f"- Bıçak skini öldürmeleri: toplam {q['knife_skin_kills_total']}, modele giren {q['knife_skin_kills_model']}. "
          f"{q['knife_note']}",
          "", "## Uyarılar"] + ([f"- ⚠️ {w}" for w in q["warnings"]] or ["- yok"])
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description="Faz 1: .dem → 15 özellik + QC (sınıflandırma YOK)")
    ap.add_argument("demo", nargs="?", help=".dem dosyası")
    ap.add_argument("--source", default="other", choices=["faceit", "hltv", "valve", "other"])
    ap.add_argument("--cs2cd", help="gerileme: CS2CD maçı (ör. no_cheater_present/108)")
    ap.add_argument("--resume", action="store_true", help="_stages/ ara çıktılarını yeniden kullan")
    ap.add_argument("--out", default=str(OUT_ROOT))
    a = ap.parse_args(argv)
    if not a.demo and not a.cs2cd:
        ap.error("demo dosyası veya --cs2cd gerekli")
    try:
        work = analyze(a.demo, cs2cd=a.cs2cd, source=a.source, out_root=Path(a.out), resume=a.resume)
    except AnalysisError as e:
        print(f"HATA: {e}", file=sys.stderr)
        return 2
    q = json.loads((work / "qc.json").read_text(encoding="utf-8"))
    m = json.loads((work / "metadata.json").read_text(encoding="utf-8"))
    print(f"Çıktı: {work}")
    print(f"Harita {q['map']} | tick {q['tickrate']} | round {q['rounds']} | oyuncu {q['n_players']} | "
          f"modele girecek öldürme {q['kills_model']} | süre {m['run']['seconds_total']} sn | tepe RAM {m['run']['peak_ram_mb']} MB")
    print(f"Model v1 için: {'UYGUN' if q['model_v1_applicable'] else 'UYGUN DEĞİL — ' + '; '.join(q['model_v1_reasons'])}")
    for w in q["warnings"]:
        print(f"  UYARI: {w}")
    return 0


if __name__ == "__main__":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    sys.exit(main())
