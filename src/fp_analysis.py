"""Yanlış pozitif (yanlış alarm) analizi — keşif amaçlı, model değiştirilmez.

Soru: "Model temiz bir oyuncuyu neden hileci sanıyor?" Rapor: MODEL_CARD.md
    python src/fp_analysis.py flags      # A ve B için oyuncu bazında karar kayıtları → data/processed/fp_flags.parquet
    python src/fp_analysis.py compare    # sayılar, GA'lar, grup karşılaştırmaları
    python src/fp_analysis.py top        # en yüksek skorlu 5 yanlış pozitifin öldürmeleri
"""
import sys

import numpy as np
import polars as pl
from sklearn.model_selection import GroupKFold

import wallhack_eval as W
from baseline import AIM_FEATURES, safe_auc

FP_RULE = 0.5     # temiz oyuncu, tekrarların ≥ %50'sinde eşiği geçtiyse "yanlış pozitif"
N_BOOT = 2000


def flags():
    """W.evaluate ile aynı katlar/eşikler; her tekrarda her oyuncunun skoru ve kararı kaydedilir."""
    from audit import team_results

    df = W.dataset(False)
    results = team_results(df["match"].unique().to_list())
    rows = []
    for model in ["A", "B"]:
        feats, imp, _ = W.MODELS[model]
        X = df.select(feats).to_numpy().astype(float)
        y, groups = df["y"].to_numpy(), df["pair"].to_numpy()
        imp_idx = [feats.index(c) for c in imp]
        for r in range(W.N_REPEATS):
            for train, test in GroupKFold(W.N_SPLITS, shuffle=True, random_state=r).split(X, y, groups):
                s = W.fit_predict(X, y, train, test, imp_idx)
                thr = W.inner_threshold(df, X, y, groups, train, imp_idx, seed=1000 + r)
                p = W.player_scores(df[test], s).with_columns(model=pl.lit(model), repeat=pl.lit(r),
                                                               thr=pl.lit(thr), flag=pl.col("score") > thr)
                rows.append(p)
            print(f"{model} tekrar {r} tamam", flush=True)
    out = pl.concat(rows).join(results, on="player", how="left")
    out.write_parquet("data/processed/fp_flags.parquet")
    print(f"{out.height} satır -> data/processed/fp_flags.parquet")


def per_player(model: str) -> pl.DataFrame:
    f = pl.read_parquet("data/processed/fp_flags.parquet").filter(pl.col("model") == model)
    return f.group_by("player").agg(pl.col("y", "pair", "match", "result").first(), pl.col("len").first(),
                                    flag_rate=pl.col("flag").mean(), score=pl.col("score").mean())


def boot_rate(p: pl.DataFrame, seed=0):
    """Oyuncuların işaretlenme oranının ortalaması (= tekrarlar üzeri FPR), çift-bootstrap %95 GA."""
    rng = np.random.default_rng(seed)
    pairs = p["pair"].unique().to_numpy()
    idx = {q: np.where(p["pair"].to_numpy() == q)[0] for q in pairs}
    v = p["flag_rate"].to_numpy()
    est = [v[np.concatenate([idx[q] for q in rng.choice(pairs, len(pairs))])].mean() for _ in range(N_BOOT)]
    return v.mean(), *np.percentile(est, [2.5, 97.5])


def player_profile() -> pl.DataFrame:
    """Oyuncu düzeyi özellik profili: öldürme özelliklerinin ortalamaları + oyun istatistikleri."""
    df = W.dataset(False)
    feats = AIM_FEATURES + W.VIS
    return df.group_by("player").agg(
        *[pl.col(c).mean().alias(c) for c in feats],
        kills=pl.len(), headshot=pl.col("headshot").mean(), sniper_pay=pl.col("is_sniper").mean(),
        pistol_pay=pl.col("is_pistol").mean(),
    )


def compare():
    profile = player_profile()
    for model in ["A", "B"]:
        p = per_player(model)
        clean = p.filter(pl.col("y") == 0)
        print(f"\n######## Model {model} ########")
        print(f"temiz oyuncu (≥3 öldürme): {clean.height} | hileci: {p.filter(pl.col('y') == 1).height}")
        for res in ["kazanan", "kaybeden", "berabere"]:
            c = clean.filter(pl.col("result") == res)
            if not c.height:
                continue
            fpr, lo, hi = boot_rate(c)
            n_fp = c.filter(pl.col("flag_rate") >= FP_RULE).height
            print(f"  temiz-{res:8s}: {c.height:3d} oyuncu | FPR (tekrar ort.) {fpr:.3f} [{lo:.3f}–{hi:.3f}] | "
                  f"yanlış pozitif (≥%50 tekrar): {n_fp}")
        tot = clean.filter(pl.col("flag_rate") >= FP_RULE).height
        print(f"  toplam yanlış pozitif temiz: {tot} / {clean.height}")

    # Karşılaştırmalar model B üzerinden
    p = per_player("B").join(profile, on="player")
    p = p.with_columns(sinif=pl.when(pl.col("y") == 1).then(pl.lit("hileci"))
                              .when(pl.col("flag_rate") >= FP_RULE).then(pl.lit("FP temiz"))
                              .otherwise(pl.lit("TN temiz")))
    cols = ["score", "kills", "headshot", "sniper_pay", "pistol_pay", "reaction_ms", "wallbang", "prefov_angle",
            "inv_angle_med", "inv_frac_lt10", "shots", "speed_last_mean", "jitter", "still_ratio", "after_move"]

    def table(a: pl.DataFrame, b: pl.DataFrame, na: str, nb: str):
        rows = []
        for c in cols:
            va, vb = a[c].drop_nulls(), b[c].drop_nulls()
            pooled = np.sqrt((va.var() + vb.var()) / 2) or np.nan
            rows.append(dict(ozellik=c, **{na: round(va.median(), 3), nb: round(vb.median(), 3)},
                             etki_d=round((va.mean() - vb.mean()) / pooled, 2)))
        print(pl.DataFrame(rows).sort("etki_d", descending=True, nulls_last=True).to_pandas().to_string(index=False))

    clean = p.filter(pl.col("y") == 0)
    print(f"\n== 1) Kazanan temiz ({clean.filter(pl.col('result') == 'kazanan').height}) vs kaybeden temiz "
          f"({clean.filter(pl.col('result') == 'kaybeden').height}) — medyanlar, etki büyüklüğü d (kazanan − kaybeden) ==")
    table(clean.filter(pl.col("result") == "kazanan"), clean.filter(pl.col("result") == "kaybeden"), "kazanan", "kaybeden")
    fp, tn = p.filter(pl.col("sinif") == "FP temiz"), p.filter(pl.col("sinif") == "TN temiz")
    print(f"\n== 2) FP temiz ({fp.height}) vs TN temiz ({tn.height}) — d = FP − TN ==")
    table(fp, tn, "FP", "TN")
    ch = p.filter(pl.col("sinif") == "hileci")
    print(f"\n== 3) Hileci ({ch.height}) vs FP temiz ({fp.height}) — d = hileci − FP ==")
    table(ch, fp, "hileci", "FP")
    print("\nFP temizlerin maç sonucu:", fp.group_by("result").len().sort("result").to_dicts())
    print("FP temizlerin harita/maç dağılımı (aynı maçtan kaç FP):",
          fp.group_by("match").len().sort("len", descending=True).head(8).to_dicts())


def top(n: str = "5"):
    """En yüksek ortalama skorlu n yanlış pozitif temiz oyuncunun öldürme pencereleri."""
    p = per_player("B").filter((pl.col("y") == 0) & (pl.col("flag_rate") >= FP_RULE)).sort("score", descending=True)
    df = W.dataset(False)
    prof = player_profile()
    for row in p.head(int(n)).iter_rows(named=True):
        pid = row["player"]
        st = prof.filter(pl.col("player") == pid).row(0, named=True)
        k = df.filter(pl.col("player") == pid).sort("tick")
        print(f"\n######## {pid} | ort. skor {row['score']:.3f}, işaretlenme {row['flag_rate']:.0%}, "
              f"takım {row['result']}, {st['kills']} öldürme, HS {st['headshot']:.0%}, sniper payı {st['sniper_pay']:.0%}")
        print(k.select("tick", "weapon", "headshot", "wallbang", pl.col("reaction_ms").round(0),
                       pl.col("prefov_angle").round(1), pl.col("inv_angle_med").round(1),
                       pl.col("inv_frac_lt10").round(2), "shots", pl.col("speed_last_mean").round(1))
               .to_pandas().to_string(index=False))


if __name__ == "__main__":
    {"flags": flags, "compare": compare, "top": top}[sys.argv[1]](*sys.argv[2:])
