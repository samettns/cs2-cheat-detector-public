"""Gün 4: her öldürme için nişan özelliklerini çıkarır.

Çıktı: data/processed/kill_features.parquet (her satır = bir öldürme)
    python src/features.py
"""
import numpy as np
import polars as pl

from aim import TICKRATE, angle_diff
from load import event_df, list_matches, load_events, load_ticks, cheaters

BEFORE = 64        # öldürmeden önceki 1 saniye
LAST = 16          # öldürmeden önceki son 250 ms
AFTER = 32         # öldürmeden sonraki 500 ms
MAX_SPEED = 3000   # °/s; bunun üstü insan hareketi değil (doğma anında bakış sıfırlaması vb.)

# Nişan gerektirmeyen öldürmeler (bıçak, el bombası, molotof...) analize girmez
# Silah sınıfı: tek atışlıklar ile otomatikler çok farklı nişan davranışı gösterir
WEAPON_CLASS = {
    "ssg08": "sniper", "awp": "sniper", "scar20": "sniper", "g3sg1": "sniper",
    "deagle": "tabanca", "revolver": "tabanca", "usp_silencer": "tabanca", "glock": "tabanca",
    "hkp2000": "tabanca", "p250": "tabanca", "fiveseven": "tabanca", "tec9": "tabanca",
    "cz75a": "tabanca", "elite": "tabanca",
}  # geri kalanlar: "otomatik" (tüfek, SMG, pompalı...)

NON_AIM_WEAPONS = {"knife", "knife_t", "bayonet", "hegrenade", "inferno", "molotov",
                   "incgrenade", "taser", "world", "planted_c4", "flashbang", "smokegrenade", "decoy"}


def player_arrays(ticks: pl.DataFrame, t0: int, n: int) -> dict:
    """Her oyuncu için tick-indeksli diziler: yaw, pitch, alive (eksik tick = NaN/False)."""
    out = {}
    for (sid,), p in ticks.group_by("steamid"):
        idx = p["tick"].to_numpy() - t0
        yaw = np.full(n, np.nan); pitch = np.full(n, np.nan); alive = np.zeros(n, bool)
        yaw[idx] = p["yaw"].to_numpy(); pitch[idx] = p["pitch"].to_numpy()
        alive[idx] = p["is_alive"].fill_null(False).to_numpy()
        d_yaw = angle_diff(yaw[1:], yaw[:-1])
        d_pitch = np.diff(pitch)
        speed = np.concatenate([[np.nan], np.hypot(d_yaw, d_pitch) * TICKRATE])
        out[sid] = dict(yaw=yaw, pitch=pitch, alive=alive, speed=speed)
    return out


def kill_features(arr: dict, k: int, shots: np.ndarray) -> dict | None:
    """Tek bir öldürme için özellikler. k = öldürme tick'inin dizideki indeksi."""
    lo, hi = k - BEFORE, k + AFTER
    if lo < 1 or hi >= len(arr["speed"]) or not arr["alive"][lo:k + 1].all():
        return None
    s = arr["speed"][lo:k + 1]
    if np.isnan(s).any() or s.max() > MAX_SPEED:
        return None
    acc = np.diff(s) * TICKRATE                       # açısal ivme (°/s²)
    after = arr["speed"][k + 1:hi + 1]
    after = after[~np.isnan(after) & (after < MAX_SPEED)]
    pitch = arr["pitch"][k - LAST:k + 1]
    return dict(
        speed_mean=s.mean(),
        speed_max=s.max(),
        speed_last_mean=s[-LAST:].mean(),             # son 250 ms ortalama hız
        jitter=np.abs(np.diff(acc)).mean() / 1e3,     # ivmedeki ani değişim (titreklik)
        still_ratio=(s < 1).mean(),                   # neredeyse hiç kıpırdamadığı tick oranı
        pitch_range_last=np.nanmax(pitch) - np.nanmin(pitch),  # son 250 ms dikey hareket (sekme kontrolü)
        shots=int(((shots >= lo) & (shots <= k)).sum()),        # son 1 sn'deki atış sayısı
        after_move=after.sum() / TICKRATE if len(after) else np.nan,  # sonraki 500 ms'de toplam dönüş (°)
    )


def match_features(match) -> list[dict]:
    ev = load_events(match)
    ch = cheaters(ev)
    has_cheater = match.parent.name == "with_cheater_present"
    ticks = load_ticks(match, ["tick", "steamid", "yaw", "pitch", "is_alive"])
    t0, n = ticks["tick"].min(), ticks["tick"].max() - ticks["tick"].min() + 1
    arrays = player_arrays(ticks, t0, n)

    fires = event_df(ev, "weapon_fire")
    shots_by = {sid: g["tick"].to_numpy() - t0 for (sid,), g in fires.group_by("user_steamid")}
    kills = event_df(ev, "player_death").filter(
        pl.col("attacker_steamid").is_not_null()
        & (pl.col("attacker_steamid") != pl.col("user_steamid"))  # intihar değil
        & ~pl.col("weapon").is_in(list(NON_AIM_WEAPONS))
    )

    rows = []
    for kill in kills.iter_rows(named=True):
        sid = kill["attacker_steamid"]
        if sid not in arrays:
            continue
        f = kill_features(arrays[sid], kill["tick"] - t0, shots_by.get(sid, np.array([])))
        if f is None:
            continue
        group = "temiz" if not has_cheater else ("hileci" if sid in ch else "belirsiz")
        rows.append(dict(match=f"{match.parent.name}/{match.name}", player=f"{match.parent.name}/{match.name}_{sid}", group=group,
                         tick=kill["tick"], weapon=kill["weapon"],
                         weapon_class=WEAPON_CLASS.get(kill["weapon"], "otomatik"), headshot=kill["headshot"], **f))
    return rows


KILL_FEATURES = "data/processed/kill_features.parquet"
VISIBILITY_TABLE = "data/processed/wallhack_features.parquet"


def attach_visibility(df: pl.DataFrame) -> pl.DataFrame:
    """5 görünürlük tabanlı sütunu (baseline.VISIBILITY_FEATURES) öldürme tablosuna ekler.

    Değerler src/wallhack_features.py'nin ürettiği tablodan AYNEN alınır (tanımlar:
    MODEL_CARD.md); burada hiçbir hesap yapılmaz, NaN'lar doldurulmaz.
    Görünürlük hesaplanmamış öldürmeler (ör. data/pairs.csv dışındaki maçlar) NaN kalır.
    Sıra: features.py (nişan) → visibility.py windows → wallhack_features.py build → features.py attach.
    """
    from pathlib import Path

    from baseline import VISIBILITY_FEATURES

    df = df.drop([c for c in VISIBILITY_FEATURES if c in df.columns])   # tekrar çalıştırmada eski sütunlar
    if not Path(VISIBILITY_TABLE).exists():
        print(f"UYARI: {VISIBILITY_TABLE} yok; görünürlük sütunları NaN.")
        return df.with_columns([pl.lit(None, dtype=pl.Float64).alias(c) for c in VISIBILITY_FEATURES])
    vis = (pl.read_parquet(VISIBILITY_TABLE)
             .select("match", "player", pl.col("kill_tick").alias("tick"), *VISIBILITY_FEATURES))
    out = df.join(vis, on=["match", "player", "tick"], how="left")
    assert out.height == df.height, "join satır sayısını değiştirmemeli"
    return out


def main():
    rows = []
    for m in list_matches():
        r = match_features(m)
        print(f"{m.parent.name}/{m.name}: {len(r)} öldürme")
        rows += r
    df = attach_visibility(pl.DataFrame(rows))
    df.write_parquet(KILL_FEATURES)
    print(f"Toplam {df.height} öldürme -> {KILL_FEATURES}")


def attach():
    """Nişan özelliklerini yeniden hesaplamadan sadece görünürlük sütunlarını ekler/günceller."""
    from baseline import VISIBILITY_FEATURES

    df = attach_visibility(pl.read_parquet(KILL_FEATURES))
    df.write_parquet(KILL_FEATURES)
    n = df.filter(pl.col(VISIBILITY_FEATURES[0]).is_not_null()).height
    print(f"{df.height} öldürme, {n} tanesinde görünürlük değeri var -> {KILL_FEATURES}")


if __name__ == "__main__":
    import sys

    {"main": main, "attach": attach}[sys.argv[1] if len(sys.argv) > 1 else "main"]()
