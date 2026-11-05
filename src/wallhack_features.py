"""Wallhack-benzeri davranış özellikleri (yöntem özeti: MODEL_CARD.md).

Girdi: data/processed/visibility.parquet + olaylar (player_death, player_hurt).
Çıktı: data/processed/wallhack_features.parquet (her satır = bir öldürme).
    python src/wallhack_features.py build
    python src/wallhack_features.py describe
"""
import sys

import numpy as np
import polars as pl

from load import RAW_DIR, event_df, load_events

TICKRATE = 64
STEP = 2                 # visibility.parquet örnek aralığı
INV_MIN_TICKS = 10       # F1: en az bu kadar "görünmez rakipli" an
INV_THRESHOLD = 10.0     # F1: derece
MASK_MAX_FRAC = 0.5      # F1: pencerede maskeli satır oranı üst sınırı
PREFOV_OFFSET = 12       # F2: t_vis − 12 tick ≈ 188 ms
KEY = ["match", "player", "kill_tick"]
from baseline import VISIBILITY_FEATURES as VIS_FEATURES  # noqa: E402  (tek kaynak)


def sid_of(player: str) -> str:
    """'klasör/maç_Player_N' → 'Player_N'."""
    return "Player_" + player.rsplit("_", 1)[-1]


def invisible_tracking(vis: pl.DataFrame) -> pl.DataFrame:
    """F1: görünmez rakiplere en yakın nişan açısının pencere özetleri."""
    masked = vis.group_by(KEY).agg(mask_frac=pl.col("visible").is_null().mean())
    per_tick = (vis.filter(pl.col("visible") == False)  # noqa: E712  (null'lar dahil değil)
                   .group_by(KEY + ["tick"]).agg(min_angle=pl.col("angle").min()))
    f1 = per_tick.group_by(KEY).agg(
        n_inv=pl.len(),
        inv_angle_med=pl.col("min_angle").median(),
        inv_angle_p90=pl.col("min_angle").quantile(0.9),
        inv_frac_lt10=(pl.col("min_angle") < INV_THRESHOLD).mean(),
    )
    f1 = masked.join(f1, on=KEY, how="left").with_columns(pl.col("n_inv").fill_null(0))
    reason = (pl.when(pl.col("mask_frac") > MASK_MAX_FRAC).then(pl.lit("maske"))
                .when(pl.col("n_inv") < INV_MIN_TICKS).then(pl.lit("az_gorunmez_an"))
                .otherwise(None))
    f1 = f1.with_columns(f1_nan=reason)
    return f1.with_columns([pl.when(pl.col("f1_nan").is_null()).then(pl.col(c)).otherwise(None).alias(c)
                            for c in ["inv_angle_med", "inv_angle_p90", "inv_frac_lt10"]])


def onset_features(vis: pl.DataFrame, deaths: dict, hurts: dict) -> pl.DataFrame:
    """F2 pre-FOV ve F4 tepki süresi: kurbanın ilk görünmez → görünür geçişine göre."""
    rows = []
    victim_rows = vis.filter(pl.col("is_victim")).sort(KEY + ["tick"])
    for (match, player, kill_tick), g in victim_rows.group_by(KEY, maintain_order=True):
        sid = sid_of(player)
        death = deaths.get((match, kill_tick, sid), {})
        victim = g["enemy"][0]
        vis_at = dict(zip(g["tick"].to_list(), g["visible"].to_list()))
        ang_at = dict(zip(g["tick"].to_list(), g["angle"].to_list()))
        row = dict(match=match, player=player, kill_tick=kill_tick, t_vis=None, prefov_angle=None,
                   reaction_ms=None, f2_nan=None)
        # Geçiş: önceki örnek görünmez (False), bu örnek görünür (True)
        t_vis = next((t for t in sorted(vis_at) if vis_at[t] is True and vis_at.get(t - STEP) is False), None)
        if death.get("thrusmoke"):
            row["f2_nan"] = "duman"
        elif t_vis is None:
            row["f2_nan"] = "gecis_yok"
        elif any(vis_at.get(t, "yok") is None for t in range(t_vis - PREFOV_OFFSET, t_vis + 1, STEP)):
            row["f2_nan"] = "maske"
        elif (t_vis - PREFOV_OFFSET) not in ang_at:
            row["f2_nan"] = "pencere_disi"
        else:
            row["t_vis"] = t_vis
            row["prefov_angle"] = ang_at[t_vis - PREFOV_OFFSET]
            hit = [t for t in hurts.get((match, sid, victim), []) if t_vis <= t <= kill_tick]
            if hit:
                row["reaction_ms"] = (min(hit) - t_vis) / TICKRATE * 1000
        rows.append(row)
    return pl.DataFrame(rows, schema_overrides={"t_vis": pl.Int64, "prefov_angle": pl.Float64,
                                                 "reaction_ms": pl.Float64, "f2_nan": pl.Utf8})


def compute(vis: pl.DataFrame, events_for) -> pl.DataFrame:
    """Görünürlük pencerelerinden (vis) 5 özelliği hesaplar. events_for(match) → o maçın olay sözlüğü.
    build() ve demo adaptörü (tek maç) bunu kullanır."""
    deaths, hurts = {}, {}
    for match in vis["match"].unique().to_list():
        ev = events_for(match)
        for d in event_df(ev, "player_death").iter_rows(named=True):
            deaths[(match, d["tick"], d["attacker_steamid"])] = d
        for h in event_df(ev, "player_hurt").iter_rows(named=True):
            hurts.setdefault((match, h["attacker_steamid"], h["user_steamid"]), []).append(h["tick"])

    keys = vis.select(KEY).unique()
    wall = keys.with_columns(wallbang=pl.struct(KEY).map_elements(
        lambda r: float((deaths.get((r["match"], r["kill_tick"], sid_of(r["player"])), {}).get("penetrated") or 0) > 0),
        return_dtype=pl.Float64))
    return (keys.join(invisible_tracking(vis), on=KEY, how="left")
                .join(onset_features(vis, deaths, hurts), on=KEY, how="left")
                .join(wall, on=KEY, how="left")
                .sort(KEY))


def build():
    vis = pl.read_parquet("data/processed/visibility.parquet")
    out = compute(vis, lambda match: load_events(RAW_DIR / match))
    out.write_parquet("data/processed/wallhack_features.parquet")
    print(f"{out.height} öldürme -> data/processed/wallhack_features.parquet")


def describe():
    """Aşama 1: silah sınıfı içinde temiz/hileci dağılımları, NaN oranları ve nedenleri."""
    from baseline import load

    kills = load().select("match", "player", pl.col("tick").alias("kill_tick"), "weapon_class", "group")
    f = pl.read_parquet("data/processed/wallhack_features.parquet").join(kills, on=KEY)  # belirsiz hariç
    print(f"{f.height} öldürme (temiz {f.filter(pl.col('group') == 'temiz').height}, "
          f"hileci {f.filter(pl.col('group') == 'hileci').height})\n")

    print("== NaN oranları (gruba göre) ==")
    print(f.group_by("group").agg(*[pl.col(c).is_null().mean().round(3).alias(c) for c in VIS_FEATURES])
           .sort("group").to_pandas().to_string(index=False))
    print("\n== Dışlanma nedenleri ==")
    print("F1:", f.group_by("group", "f1_nan").len().sort("group", "f1_nan").to_dicts())
    print("F2/F4:", f.group_by("group", "f2_nan").len().sort("group", "f2_nan").to_dicts())
    hurt_missing = f.filter(pl.col("f2_nan").is_null() & pl.col("reaction_ms").is_null()).height
    print(f"F4 ek: geçiş var ama t_vis sonrası hasar yok: {hurt_missing}")

    for feat in ["inv_angle_med", "inv_frac_lt10", "inv_angle_p90", "prefov_angle", "reaction_ms"]:
        print(f"\n== {feat} (silah sınıfı içinde) ==")
        print(f.drop_nulls(feat).group_by("weapon_class", "group")
               .agg(pl.len().alias("n"), pl.col(feat).quantile(0.25).round(2).alias("q25"),
                    pl.col(feat).median().round(2).alias("medyan"), pl.col(feat).quantile(0.75).round(2).alias("q75"))
               .sort("weapon_class", "group").to_pandas().to_string(index=False))
    print("\n== wallbang oranı (silah sınıfı içinde; öldürme ve oyuncu düzeyi) ==")
    print(f.group_by("weapon_class", "group").agg(pl.len().alias("n"), pl.col("wallbang").mean().round(3).alias("öldürme"))
           .join(f.group_by("weapon_class", "group", "player").agg(pl.col("wallbang").mean())
                  .group_by("weapon_class", "group").agg(pl.col("wallbang").median().round(3).alias("oyuncu_medyan")),
                 on=["weapon_class", "group"])
           .sort("weapon_class", "group").to_pandas().to_string(index=False))


def plot():
    """Aşama 1 grafiği: 4 özellik × 3 silah sınıfı, temiz vs hileci (wallbang için oran çubuğu)."""
    import matplotlib.pyplot as plt

    from baseline import load
    from plot_aim import CHEAT_COLOR, CLEAN_COLOR, INK, MUTED, style

    kills = load().select("match", "player", pl.col("tick").alias("kill_tick"), "weapon_class", "group")
    f = pl.read_parquet("data/processed/wallhack_features.parquet").join(kills, on=KEY)
    groups = [("temiz", CLEAN_COLOR), ("hileci", CHEAT_COLOR)]
    classes = ["sniper", "tabanca", "otomatik"]
    feats = [("inv_angle_med", "görünmezken en yakın rakibe\nnişan açısı, medyan (°)"),
             ("prefov_angle", "görünür olmadan ~190 ms önce\nkurbana nişan açısı (°)"),
             ("reaction_ms", "görünür olunca ilk\nisabete kadar süre (ms)")]
    fig, axes = plt.subplots(len(feats) + 1, 3, figsize=(11, 12))
    for c, wc in enumerate(classes):
        sub = f.filter(pl.col("weapon_class") == wc)
        for r, (feat, label) in enumerate(feats):
            ax = axes[r, c]
            data = [sub.filter(pl.col("group") == g)[feat].drop_nulls().to_numpy() for g, _ in groups]
            bp = ax.boxplot(data, widths=0.5, showfliers=False, patch_artist=True,
                            medianprops=dict(color=INK, linewidth=2))
            for patch, (_, color) in zip(bp["boxes"], groups):
                patch.set(facecolor=color, alpha=0.75, edgecolor=color)
            for el in ("whiskers", "caps"):
                for line in bp[el]:
                    line.set_color(MUTED)
            ax.set_xticks([1, 2], [f"{g}\n(n={len(d)})" for (g, _), d in zip(groups, data)])
            style(ax)
            if c == 0:
                ax.set_ylabel(label, color=MUTED, fontsize=9)
            if r == 0:
                ax.set_title(wc, color=INK, fontsize=12, loc="left")
        ax = axes[-1, c]
        rates = [sub.filter(pl.col("group") == g)["wallbang"].mean() for g, _ in groups]
        ax.bar([1, 2], rates, width=0.5, color=[col for _, col in groups], alpha=0.85)
        for x, v in zip([1, 2], rates):
            ax.text(x, v + 0.01, f"%{v * 100:.0f}", ha="center", color=INK, fontsize=10)
        ax.set_xticks([1, 2], [g for g, _ in groups])
        ax.set_ylim(0, 0.7)
        style(ax)
        if c == 0:
            ax.set_ylabel("duvar delen öldürme oranı", color=MUTED, fontsize=9)
    fig.suptitle(f"Wallhack-benzeri davranış özellikleri — silah sınıfı içinde (50 çift, {f.height} öldürme)",
                 color=INK, fontsize=13, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig("reports/figures/wallhack_features.png", dpi=150)
    print("Kaydedildi: reports/figures/wallhack_features.png")


if __name__ == "__main__":
    {"build": build, "describe": describe, "plot": plot}[sys.argv[1]]()
