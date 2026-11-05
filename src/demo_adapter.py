"""Güncel CS2 .dem → CS2CD formatı (demoparser2). Model/özellik/eşik DEĞİŞTİRMEZ; sadece veri dönüştürür.

Çıktı (git dışı, data/raw/ altında):
    data/raw/current_converted/<kaynak>/<kaynak>_<n>_<harita>.parquet   tick tablosu (CS2CD sütun adları)
    data/raw/current_converted/<kaynak>/<kaynak>_<n>_<harita>.json      olaylar + CSstats_info + demo_meta
    data/raw/current_converted/<kaynak>/<...>.ids.json                  anonim ↔ gerçek kimlik eşlemesi (YEREL, paylaşma)

Gizlilik: Steam ID'ler demo içinde Player_1..N ile değiştirilir; isim sütunları tamamen atılır.
Sadece iki takımdaki oyuncular tutulur (team_num 2/3): izleyici/yorumcu (HLTV) rakip sayılmasın.
"""
import json
from pathlib import Path

import polars as pl
from demoparser2 import DemoParser

CURRENT_DIR = Path("data/raw/current")
CONVERTED_DIR = Path("data/raw/current_converted")
SOURCES = ["faceit", "hltv", "valve"]

# Hattın (nişan + görünürlük + model) ve kontrollerin kullandığı tick alanları (CS2CD adlarıyla)
TICK_PROPS = ["X", "Y", "Z", "yaw", "pitch", "is_alive", "team_num", "health", "duck_amount", "last_place_name",
              "approximate_spotted_by", "spotted", "is_scoped", "zoom_lvl", "active_weapon_name", "FIRE",
              "usercmd_mouse_dx", "usercmd_mouse_dy", "team_rounds_total", "game_time",
              # Faz 1 bağlamı (model kullanmaz): round numarası, ısınma, skor tablosu (maç başlangıcı + tutarlılık testi)
              "total_rounds_played", "is_warmup_period", "kills_total", "deaths_total",
              # Faz 3 bağlamı: round sınırları donma süresi bitişlerine göre (round/aşama etiketi; model kullanmaz)
              "is_freeze_period"]
# Model için zorunlu olaylar ve alanları
REQUIRED_EVENTS = {
    "player_death": ["tick", "attacker_steamid", "user_steamid", "weapon", "headshot", "penetrated", "thrusmoke"],
    "weapon_fire": ["tick", "user_steamid", "weapon"],
    "player_hurt": ["tick", "attacker_steamid", "user_steamid"],
}
# Raporlama/görselleştirme/doğrulama için (yoksa model etkilenmez)
OPTIONAL_EVENTS = ["bullet_damage", "round_freeze_end", "round_officially_ended", "round_start", "round_end",
                   "smokegrenade_detonate", "smokegrenade_expired", "inferno_startburn", "inferno_expire",
                   "player_blind", "bomb_planted"]


def demo_files() -> list[tuple[str, Path]]:
    return [(src, f) for src in SOURCES for f in sorted((CURRENT_DIR / src).glob("*.dem"))]


def convert(source: str, path: Path, index: int, out_dir: Path | None = None, match_id: str | None = None) -> dict:
    """Bir demoyu dönüştürür; meta bilgi sözlüğü döndürür (anonim).
    out_dir/match_id verilmezse Faz 0 düzeni: data/raw/current_converted/<kaynak>/<kaynak>_<n>_<harita>."""
    parser = DemoParser(str(path))
    header = parser.parse_header()
    map_name = header.get("map_name")
    match_id = match_id or f"{source}_{index}_{map_name}"
    out_dir = out_dir or CONVERTED_DIR / source
    out_dir.mkdir(parents=True, exist_ok=True)

    ticks = pl.from_pandas(parser.parse_ticks(TICK_PROPS))
    present_props = [c for c in TICK_PROPS if c in ticks.columns]
    ticks = ticks.with_columns(pl.col("steamid").cast(pl.Utf8))
    all_rows = ticks.height
    ticks = ticks.filter(pl.col("team_num").is_in([2, 3]))
    # Anonim kimlik: oyuncuların ilk görüldüğü tick, eşitlikte steamid sırası (deterministik)
    order = (ticks.group_by("steamid").agg(pl.col("tick").min().alias("first")).sort("first", "steamid"))
    ids = {sid: f"Player_{i + 1}" for i, sid in enumerate(order["steamid"].to_list())}
    ticks = (ticks.with_columns(pl.col("steamid").replace_strict(ids))
                  .with_columns(pl.col("approximate_spotted_by").list.eval(
                      pl.element().cast(pl.Utf8).replace_strict(ids, default=None)).list.drop_nulls())
                  .drop([c for c in ticks.columns if c == "name"]))
    ticks.write_parquet(out_dir / f"{match_id}.parquet")

    available = set(parser.list_game_events())
    events, event_fields = {}, {}
    for name in list(REQUIRED_EVENTS) + OPTIONAL_EVENTS:
        if name not in available:
            continue
        df = pl.from_pandas(parser.parse_event(name))
        df = df.drop([c for c in df.columns if c.endswith("_name") or c.endswith("_xuid")])
        for c in [c for c in df.columns if c.endswith("_steamid")]:
            df = df.with_columns(pl.col(c).cast(pl.Utf8).replace_strict(ids, default=None))
        event_fields[name] = df.columns
        events[name] = df.to_dicts()
    events["CSstats_info"] = [dict(map=map_name, server=source, avg_rank=None, match_making_type=source)]
    events["cheaters"] = []
    events["demo_meta"] = [dict(source=source, patch_version=header.get("patch_version"),
                                demo_version_name=header.get("demo_version_name"))]
    with open(out_dir / f"{match_id}.json", "w", encoding="utf-8") as f:
        json.dump(events, f, default=str)
    with open(out_dir / f"{match_id}.ids.json", "w", encoding="utf-8") as f:   # yerel, git dışı
        json.dump({v: k for k, v in ids.items()}, f)

    return dict(match_id=match_id, source=source, map=map_name, patch_version=header.get("patch_version"),
                demo_version_name=header.get("demo_version_name"), tick_props_present=present_props,
                tick_props_missing=[c for c in TICK_PROPS if c not in present_props],
                rows_all=all_rows, rows_players=ticks.height, n_players=len(ids),
                events_available=sorted(available), event_fields=event_fields)
