"""CS2CD maçlarını yüklemek için yardımcı fonksiyonlar.

Bir maç = <id>.parquet (her tick'te her oyuncu için bir satır) + <id>.json (olaylar, etiketler).
"""
import json
from pathlib import Path

import polars as pl

RAW_DIR = Path("data/raw")

# Şimdilik ihtiyacımız olan sütunlar. Parquet sütun bazlı olduğu için
# sadece bunları okumak tüm dosyayı okumaktan çok daha hızlı.
TICK_COLUMNS = [
    "tick", "steamid", "team_num", "is_alive", "health",
    "X", "Y", "Z", "pitch", "yaw",
    "usercmd_mouse_dx", "usercmd_mouse_dy",
    "FIRE", "active_weapon_name", "spotted",
]


def list_matches() -> list[Path]:
    """İndirilmiş tüm maçların yolunu (uzantısız) döndürür."""
    return sorted(p.with_suffix("") for p in RAW_DIR.glob("*/*.parquet"))


def load_events(match: Path) -> dict:
    with open(match.with_suffix(".json"), encoding="utf-8") as f:
        return json.load(f)


def load_ticks(match: Path, columns: list[str] | None = TICK_COLUMNS) -> pl.DataFrame:
    return pl.read_parquet(match.with_suffix(".parquet"), columns=columns)


def cheaters(events: dict) -> set[str]:
    """Maçta hileci olarak etiketlenmiş oyuncuların (anonim) steamid'leri."""
    return {c["steamid"] for c in events.get("cheaters", [])}


def event_df(events: dict, name: str) -> pl.DataFrame:
    """JSON'daki bir olay listesini (örn. 'player_death') tabloya çevirir."""
    return pl.DataFrame(events.get(name, []))
