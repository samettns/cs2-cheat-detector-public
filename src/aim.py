"""Nişan (aim) hareketi hesapları."""
import numpy as np
import polars as pl

TICKRATE = 64


def angle_diff(a, b):
    """a - b farkını -180..+180 aralığına sarar.

    Yaw 179° -> -179° geçişinde ham fark -358° çıkar, ama gerçek hareket +2°'dir.
    (fark + 180) % 360 - 180 bu sıçramayı düzeltir.
    """
    return (a - b + 180) % 360 - 180


def add_aim_speed(ticks: pl.DataFrame) -> pl.DataFrame:
    """Her oyuncu için tick'ten tick'e nişan değişimini ve açısal hızı (derece/saniye) ekler.

    Beklenen sütunlar: steamid, tick, yaw, pitch.
    """
    ticks = ticks.sort("steamid", "tick")
    d_yaw = angle_diff(pl.col("yaw"), pl.col("yaw").shift(1)).over("steamid")
    d_pitch = (pl.col("pitch") - pl.col("pitch").shift(1)).over("steamid")
    ticks = ticks.with_columns(d_yaw=d_yaw, d_pitch=d_pitch)
    return ticks.with_columns(
        aim_speed=(pl.col("d_yaw") ** 2 + pl.col("d_pitch") ** 2).sqrt() * TICKRATE
    )


def kill_windows(ticks: pl.DataFrame, kills: pl.DataFrame, player: str,
                 before: int = 64, after: int = 16) -> np.ndarray:
    """Oyuncunun her öldürmesi etrafındaki aim_speed penceresini döndürür.

    Satırlar = öldürmeler, sütunlar = öldürmeye göre tick (-before .. +after).
    Pencere maç dışına taşarsa veya oyuncu o arada ölüyse o öldürme atlanır.
    """
    p = ticks.filter(pl.col("steamid") == player).sort("tick")
    speed = dict(zip(p["tick"].to_list(), p["aim_speed"].to_list()))
    alive = dict(zip(p["tick"].to_list(), p["is_alive"].to_list()))
    rows = []
    for k in kills.filter(pl.col("attacker_steamid") == player)["tick"].to_list():
        rng = range(k - before, k + after + 1)
        if all(t in speed and speed[t] is not None and alive[t] for t in rng):
            rows.append([speed[t] for t in rng])
    return np.array(rows)
