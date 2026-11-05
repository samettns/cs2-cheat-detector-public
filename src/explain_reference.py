"""Faz 2 açıklama ve olay eşiği referansı (model v1'den TÜRETİLMİŞ; model/özellik/eşik DEĞİŞMEZ).

Tanımlar: MODEL_CARD.md (İnceleme önceliği; kod ve demo sonuçlarından önce yazıldı).

Çıktı: data/reference/v1/  (git'te; yalnız anonim CS2CD eğitim verisinden türetilmiş dağılımlar, kimlik yok)
    kill_oof_clean.parquet      temiz oyuncuların OOF öldürme skorları + stil (olay eşiği ve olay yüzdeliği için)
    oof_all_kills.parquet       değerlendirmedeki bütün öldürmelerin OOF skorları (eğitim maçlarının gösterimi)
    clean_kill_features.parquet temiz eğitim öldürmeleri: stil, silah sınıfı, 13 açıklama özelliği
    clean_player_features.parquet temiz oyuncuların (>= 3 öldürme) öldürme ortalamaları + stil
    reference.json              olay eşikleri (stil başına), "hileci yönü" işaretleri, tutarlılık kontrolü, kaynak hash'leri

    python src/explain_reference.py
OOF protokolü freeze.py / holdout.oof_players ile AYNI (havuzlu 160 çift, çift GroupKFold 5 kat × 10 tekrar,
tekrar-ortalama). Kontrol: OOF öldürme skorları oyuncu düzeyinde birleştirilince models/v1/clean_reference.parquet
birebir çıkmalı (çıkmazsa betik durur).
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import polars as pl
from polars.testing import assert_frame_equal
from sklearn.model_selection import GroupKFold

import wallhack_eval as W
from baseline import MIN_KILLS, make_model
from pooled import PAIRS_ALL
from style_threshold import STYLES, player_styles

OUT = Path("data/reference/v1")
V1 = Path("models/v1")
EVENT_Q = 0.95                       # stil içi temiz OOF öldürme skorlarının %95'lik dilimi
EXPLAIN_FEATURES = ["speed_mean", "speed_max", "speed_last_mean", "jitter", "still_ratio", "pitch_range_last",
                    "shots", "after_move", "inv_angle_med", "inv_frac_lt10", "prefov_angle", "reaction_ms"]
# wallbang: bağlam olarak gösterilir (ikili); is_sniper/is_pistol açıklamaya girmez
WEAPON_CLASSES = ["otomatik", "tabanca", "sniper"]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def oof_kill_scores(df: pl.DataFrame, feats: list[str]) -> np.ndarray:
    """holdout.oof_players ile aynı döngü; öldürme düzeyinde döndürür."""
    X = df.select(feats).to_numpy().astype(float)
    y, groups = df["y"].to_numpy(), df["pair"].to_numpy()
    oof = np.zeros(len(y))
    for r in range(W.N_REPEATS):
        for train, test in GroupKFold(W.N_SPLITS, shuffle=True, random_state=r).split(X, y, groups):
            assert not set(groups[train]) & set(groups[test])
            oof[test] += make_model().fit(X[train], y[train]).predict_proba(X[test])[:, 1]
    return oof / W.N_REPEATS


def direction(cheat: pl.Series, clean: pl.Series) -> int:
    """+1: hilecilerde ortalama daha yüksek (yüksek değer dikkat çekici); -1: daha düşük; 0: fark yok."""
    d = cheat.drop_nulls().mean() - clean.drop_nulls().mean()
    return 0 if d is None or d == 0 else (1 if d > 0 else -1)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    feats = json.loads((V1 / "features.json").read_text(encoding="utf-8"))["features"]
    df = W.dataset(pairs_files=PAIRS_ALL)
    styles = player_styles(df)
    df = df.join(styles, on="player", how="left", maintain_order="left")

    oof = oof_kill_scores(df, feats)
    # Tutarlılık: oyuncu düzeyinde birleştirince clean_reference ile birebir aynı mı?
    players = W.player_scores(df, oof).join(styles, on="player")
    mine = players.filter(pl.col("y") == 0).select("player", "stil", "score").sort("stil", "score")
    ref = pl.read_parquet(V1 / "clean_reference.parquet")
    assert_frame_equal(mine, ref, check_exact=True)
    print("Tutarlılık: OOF öldürme skorları → oyuncu skorları = clean_reference.parquet (birebir)")

    k = df.with_columns(score=pl.Series(oof))
    # eğitim havuzundaki maçların arayüzde OOF skorlarıyla gösterimi için BÜTÜN değerlendirme öldürmeleri
    # (temiz + hileci; "belirsiz" değerlendirmede yoktu → burada da yok). Kimlik: CS2CD anonim Player_N.
    k.select("match", pl.col("player").alias("player_full"),
             player=pl.lit("Player_") + pl.col("player").str.split("_").list.last(),
             tick="tick", weapon="weapon", weapon_class="weapon_class", y="y", stil="stil", pair="pair",
             score="score").sort("match", "player", "tick", "score").write_parquet(OUT / "oof_all_kills.parquet")
    # Oyuncu skoru alan (>= 3 öldürme) temiz oyuncuların öldürmeleri — clean_reference ile aynı evren
    eligible = set(players["player"].to_list())
    clean_k = k.filter((pl.col("y") == 0) & pl.col("player").is_in(list(eligible)))
    clean_k.select("stil", "weapon_class", "score").sort("stil", "weapon_class", "score").write_parquet(
        OUT / "kill_oof_clean.parquet")
    clean_k.select("stil", "weapon_class", *EXPLAIN_FEATURES, "wallbang").write_parquet(OUT / "clean_kill_features.parquet")
    pm = (k.filter(pl.col("player").is_in(list(eligible))).group_by("player", maintain_order=True)
           .agg(pl.col("y").first(), pl.col("stil").first(), *[pl.col(c).mean() for c in EXPLAIN_FEATURES + ["wallbang"]]))
    pm.filter(pl.col("y") == 0).drop("player", "y").write_parquet(OUT / "clean_player_features.parquet")

    event_thr = {st: float(np.quantile(clean_k.filter(pl.col("stil") == st)["score"].to_numpy(), EVENT_Q))
                 for st in STYLES}
    event_thr_global = float(np.quantile(clean_k["score"].to_numpy(), EVENT_Q))
    kill_dir = {wc: {f: direction(k.filter((pl.col("weapon_class") == wc) & (pl.col("y") == 1))[f],
                                  k.filter((pl.col("weapon_class") == wc) & (pl.col("y") == 0))[f])
                     for f in EXPLAIN_FEATURES} for wc in WEAPON_CLASSES}
    player_dir = {st: {f: direction(pm.filter((pl.col("stil") == st) & (pl.col("y") == 1))[f],
                                    pm.filter((pl.col("stil") == st) & (pl.col("y") == 0))[f])
                       for f in EXPLAIN_FEATURES} for st in STYLES}
    json.dump(dict(
        version="v1", definitions="MODEL_CARD.md (İnceleme önceliği)",
        oof_protocol="pooled 160 pairs, pair GroupKFold 5x10, repeat-mean (= freeze.py)",
        consistency_check="OOF kill scores aggregated (mean, >=3 kills) == models/v1/clean_reference.parquet: exact",
        event_quantile=EVENT_Q, event_threshold=event_thr, event_threshold_global=event_thr_global,
        event_reference_n={st: clean_k.filter(pl.col("stil") == st).height for st in STYLES},
        kill_direction=kill_dir, player_direction=player_dir, explain_features=EXPLAIN_FEATURES,
        min_kills=MIN_KILLS,
        sources={p.name: sha256(p) for p in [V1 / "model_B.joblib", V1 / "features.json", V1 / "thresholds.json",
                                             V1 / "clean_reference.parquet", Path("data/processed/kill_features.parquet")]},
    ), open(OUT / "reference.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("Olay eşikleri:", {k_: round(v, 3) for k_, v in event_thr.items()})
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
