"""Baseline model ve değerlendirme araçları.

Öldürme bazında sınıflandırıcı eğitir, maç bazında (GroupKFold) çapraz doğrulama yapar,
öldürme skorlarını oyuncu bazında birleştirir.
    python src/baseline.py      # özellik ablasyonu
"""
import numpy as np
import polars as pl
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

AIM_FEATURES = ["speed_mean", "speed_max", "speed_last_mean", "jitter",
                "still_ratio", "pitch_range_last", "shots", "after_move"]
WEAPON_FEATURES = ["is_sniper", "is_pistol"]
# Görünürlük tabanlı özellikler (tanımlar: MODEL_CARD.md; hesap: src/wallhack_features.py;
# features.py hattına `attach_visibility` ile eklenir). NaN'lar NaN kalır (doldurma yok).
VISIBILITY_FEATURES = ["inv_angle_med", "inv_frac_lt10", "prefov_angle", "wallbang", "reaction_ms"]

FEATURE_SETS = {
    "nişan + silah":            AIM_FEATURES + WEAPON_FEATURES,
    "sadece nişan":             AIM_FEATURES,
    "sadece silah":             WEAPON_FEATURES,
    "sadece headshot":          ["headshot"],
    "nişan + silah + headshot": AIM_FEATURES + WEAPON_FEATURES + ["headshot"],
    "nişan + silah + görünürlük": AIM_FEATURES + WEAPON_FEATURES + VISIBILITY_FEATURES,
}

N_SPLITS = 5
N_REPEATS = 10      # farklı maç bölmeleriyle tekrar → std
N_BOOT = 1000       # maç bazında bootstrap → güven aralığı
MIN_KILLS = 3       # oyuncu skoruna girmek için en az öldürme


def load(path="data/processed/kill_features.parquet"):
    df = pl.read_parquet(path)
    # "belirsiz" grup dışarıda: hileli maçlardaki "temiz" etiketler güvenilir değil
    df = df.filter(pl.col("group") != "belirsiz")
    return df.with_columns(
        y=(pl.col("group") == "hileci").cast(pl.Int8),
        headshot=pl.col("headshot").cast(pl.Int8),
        is_sniper=(pl.col("weapon_class") == "sniper").cast(pl.Int8),
        is_pistol=(pl.col("weapon_class") == "tabanca").cast(pl.Int8),
    )


def make_model():
    return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200,
                                          class_weight="balanced", random_state=0)


def aggregate(df, scores, method="mean", min_kills=MIN_KILLS, k=3, group_col="match"):
    """Öldürme skorlarını oyuncu skoruna çevirir. Birincil yöntem: mean."""
    agg = {
        "mean": pl.col("score").mean(),
        "median": pl.col("score").median(),
        "max": pl.col("score").max(),
        f"top{k}_mean": pl.col("score").sort(descending=True).head(k).mean(),
    }[method]
    cols = ["player", "match", "y"] + ([group_col] if group_col != "match" else [])
    return (df.select(cols).with_columns(score=pl.Series(scores))
              .group_by("player", maintain_order=True).agg(*[pl.col(c).first() for c in cols[1:]], agg.alias("score"), pl.len())
              .filter(pl.col("len") >= min_kills))


def safe_auc(y, s):
    y = np.asarray(y)
    return roc_auc_score(y, s) if 0 < y.sum() < len(y) else np.nan


def evaluate(df, features, n_repeats=N_REPEATS, agg="mean", min_kills=MIN_KILLS, n_boot=N_BOOT,
             group_col="match"):
    """Tekrarlı maç bazlı CV.

    Döndürür: fold AUC'lerinin ortalaması/std'si (öldürme ve oyuncu) ve
    tekrarlar boyunca ortalanmış OOF skorlar üzerinden maç-bootstrap %95 güven aralığı.
    group_col: bölme ve bootstrap birimi ("match" veya eşleşmeli veride "pair"; bir çift asla bölünmez).
    """
    X = df.select(features).to_numpy().astype(float)
    y = df["y"].to_numpy()
    groups = df[group_col].to_numpy()
    kill_aucs, player_aucs = [], []
    oof_sum = np.zeros(len(y))
    for r in range(n_repeats):
        oof = np.zeros(len(y))
        for train, test in GroupKFold(N_SPLITS, shuffle=True, random_state=r).split(X, y, groups):
            assert not set(groups[train]) & set(groups[test])  # sızıntı koruması
            model = make_model().fit(X[train], y[train])
            oof[test] = model.predict_proba(X[test])[:, 1]
            kill_aucs.append(safe_auc(y[test], oof[test]))
            p = aggregate(df[test], oof[test], agg, min_kills, group_col=group_col)
            player_aucs.append(safe_auc(p["y"], p["score"]))
        oof_sum += oof
    oof = oof_sum / n_repeats
    players = aggregate(df, oof, agg, min_kills, group_col=group_col)
    return dict(
        kill_mean=np.nanmean(kill_aucs), kill_std=np.nanstd(kill_aucs),
        player_mean=np.nanmean(player_aucs), player_std=np.nanstd(player_aucs),
        kill_ci=bootstrap_ci(groups, y, oof, n_boot),
        player_ci=bootstrap_ci(players[group_col].to_numpy(), players["y"].to_numpy(),
                               players["score"].to_numpy(), n_boot),
        n_kills=len(y), n_players=players.height, n_matches=df["match"].n_unique(),
        oof=oof,
    )


def bootstrap_ci(matches, y, s, n_boot=N_BOOT, seed=0):
    """Maçları yerine koyarak yeniden örnekler (aynı maçın satırları birlikte hareket eder)."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(matches)
    idx_by_match = {m: np.where(matches == m)[0] for m in uniq}
    aucs = []
    for _ in range(n_boot):
        idx = np.concatenate([idx_by_match[m] for m in rng.choice(uniq, len(uniq))])
        aucs.append(safe_auc(y[idx], s[idx]))
    return tuple(np.nanpercentile(aucs, [2.5, 97.5]))


def fmt(r):
    return (f"öldürme {r['kill_mean']:.3f} ± {r['kill_std']:.3f} [{r['kill_ci'][0]:.2f}–{r['kill_ci'][1]:.2f}]   "
            f"oyuncu {r['player_mean']:.3f} ± {r['player_std']:.3f} [{r['player_ci'][0]:.2f}–{r['player_ci'][1]:.2f}]")


def main():
    df = load()
    print(f"{df.height} öldürme, {df['player'].n_unique()} oyuncu, {df['match'].n_unique()} maç "
          f"(hileci öldürme oranı %{df['y'].mean() * 100:.0f})")
    print(f"AUC: fold ortalaması ± std [maç-bootstrap %95 GA], {N_REPEATS}x{N_SPLITS} GroupKFold\n")
    for name, feats in FEATURE_SETS.items():
        print(f"{name:26s} {fmt(evaluate(df, feats))}")


if __name__ == "__main__":
    main()
