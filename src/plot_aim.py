"""Gün 3: hileci ve temiz oyuncunun nişan hareketini karşılaştıran grafikler.

Çıktılar reports/figures/ altına kaydedilir.
    python src/plot_aim.py
"""
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from aim import TICKRATE, add_aim_speed, kill_windows
from load import RAW_DIR, event_df, load_events, load_ticks

CLEAN_COLOR = "#2a78d6"   # mavi
CHEAT_COLOR = "#eb6834"   # turuncu
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e6e5df"

# Adil karşılaştırma: hilecisiz maçların en çok öldüren oyuncusu vs. bariz bir hileci
PLAYERS = [
    ("Temiz – en iyi oyuncu (maç 349, Player_7)", RAW_DIR / "no_cheater_present/349", "Player_7", CLEAN_COLOR),
    ("Hileci (maç 42, Player_3)", RAW_DIR / "with_cheater_present/42", "Player_3", CHEAT_COLOR),
]
BEFORE, AFTER = 64, 16


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def load_player(match, player):
    ticks = add_aim_speed(load_ticks(match, ["tick", "steamid", "yaw", "pitch", "is_alive"]))
    kills = event_df(load_events(match), "player_death")
    return ticks, kills


def plot_kill_windows(data):
    """Her öldürmeden önceki 1 saniyedeki nişan hızı, öldürme anına hizalanmış."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    x = np.arange(-BEFORE, AFTER + 1) / TICKRATE * 1000  # milisaniye
    for ax, (title, ticks, kills, player, color) in zip(axes, data):
        w = kill_windows(ticks, kills, player, BEFORE, AFTER)
        for row in w:
            ax.plot(x, row, color=color, alpha=0.18, linewidth=1)
        ax.plot(x, np.median(w, axis=0), color=color, linewidth=2.5, label="medyan")
        ax.axvline(0, color=MUTED, linewidth=1, linestyle="--")
        ax.text(5, 590, "öldürme", color=MUTED, fontsize=9, va="top")
        ax.set_title(f"{title}\n{len(w)} öldürme", color=INK, fontsize=11, loc="left")
        ax.set_xlabel("öldürmeye göre zaman (ms)", color=MUTED)
        ax.set_ylim(0, 600)
        style(ax)
        ax.legend(frameon=False, fontsize=9, loc="upper left")
    axes[0].set_ylabel("nişan hızı (°/s)", color=MUTED)
    fig.suptitle("Öldürmeden önceki 1 saniyede nişan hızı (ince çizgi = tek öldürme)",
                 color=INK, fontsize=13, x=0.01, ha="left")
    fig.tight_layout()
    return fig


def plot_single_kill(data):
    """Tek bir öldürme etrafında yaw ve pitch'in zamana göre değişimi."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 6), sharex=True)
    for col, (title, ticks, kills, player, color) in enumerate(data):
        k_ticks = kills.filter(pl.col("attacker_steamid") == player)["tick"].to_list()
        k = k_ticks[len(k_ticks) // 2]  # ortadaki öldürme (rastgele seçim değil, tekrarlanabilir)
        p = ticks.filter((pl.col("steamid") == player) & pl.col("tick").is_between(k - 3 * TICKRATE, k + TICKRATE))
        t = (p["tick"].to_numpy() - k) / TICKRATE
        # Yaw'ı sürekli hale getir (sarma düzeltmesi): -180/180 sıçraması grafikte dikey çizgi yapmasın
        yaw = np.degrees(np.unwrap(np.radians(p["yaw"].to_numpy())))
        for row, (vals, name) in enumerate([(yaw, "yaw (°)"), (p["pitch"].to_numpy(), "pitch (°)")]):
            ax = axes[row, col]
            ax.plot(t, vals, color=color, linewidth=2)
            for kt in k_ticks:  # pencereye düşen diğer öldürmeler de
                if -3 <= (kt - k) / TICKRATE <= 1:
                    ax.axvline((kt - k) / TICKRATE, color=MUTED, linewidth=1, linestyle="--")
            ax.set_ylabel(name, color=MUTED)
            style(ax)
        axes[0, col].set_title(f"{title}\nöldürme tick {k}", color=INK, fontsize=11, loc="left")
        axes[1, col].set_xlabel("öldürmeye göre zaman (s)  — kesikli çizgi = öldürme", color=MUTED)
    fig.suptitle("Tek bir öldürme etrafında bakış açısı", color=INK, fontsize=13, x=0.01, ha="left")
    fig.tight_layout()
    return fig


def main():
    data = []
    for title, match, player, color in PLAYERS:
        ticks, kills = load_player(match, player)
        data.append((title, ticks, kills, player, color))

    plot_kill_windows(data).savefig("reports/figures/gun03_kill_windows.png", dpi=150)
    plot_single_kill(data).savefig("reports/figures/gun03_single_kill.png", dpi=150)
    print("Kaydedildi: reports/figures/gun03_*.png")


if __name__ == "__main__":
    main()
