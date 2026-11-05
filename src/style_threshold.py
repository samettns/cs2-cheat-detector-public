"""Stil bazlı karar eşiği (G1) vs tek global eşik (G0). Yöntem özeti: MODEL_CARD.md.

Model B sabit; G0 ve G1 aynı katlarda aynı skorları kullanır, sadece eşik kuralı farklıdır. Pipeline değiştirilmez.
    python src/style_threshold.py run [s1]   # karar kayıtları → data/processed/style_flags[_s1].parquet
    python src/style_threshold.py report [s1]
"""
import sys

import numpy as np
import polars as pl
from sklearn.model_selection import GroupKFold

import wallhack_eval as W
from baseline import safe_auc
from sensitivity import S1_MATCHES

STYLES = ["sniper ağırlıklı", "otomatik ağırlıklı", "diğer"]
STYLE_SHARE = 0.5
MIN_CLEAN = 40          # yedek kural: eğitim katında bir stilde < 40 temiz oyuncu → global eşik
TARGET_FPR = W.TARGET_FPR
N_BOOT = 2000


def player_styles(df: pl.DataFrame) -> pl.DataFrame:
    """Stil yalnızca silah kullanımından (etiket kullanılmaz)."""
    s = df.group_by("player").agg(sn=pl.col("is_sniper").mean(), oto=(pl.col("weapon_class") == "otomatik").mean())
    return s.select("player", stil=pl.when(pl.col("sn") > STYLE_SHARE).then(pl.lit(STYLES[0]))
                                  .when(pl.col("oto") > STYLE_SHARE).then(pl.lit(STYLES[1]))
                                  .otherwise(pl.lit(STYLES[2])))


def inner_scores(df, X, y, groups, train, seed):
    """wallhack_eval.inner_threshold ile aynı iç CV; oyuncu düzeyi iç skorları döndürür."""
    inner = np.zeros(len(train))
    Xt, yt, gt = X[train], y[train], groups[train]
    for itr, ite in GroupKFold(W.INNER_SPLITS, shuffle=True, random_state=seed).split(Xt, yt, gt):
        inner[ite] = W.fit_predict(Xt, yt, itr, ite, [])
    return W.player_scores(df[train], inner)


def dataset(variant: str) -> pl.DataFrame:
    df = W.dataset(False)
    if variant == "s1":
        pairs = pl.read_csv("data/pairs.csv")
        drop = pairs.filter(pl.col("clean").is_in(S1_MATCHES) | pl.col("cheat").is_in(S1_MATCHES))["pair"].to_list()
        df = df.filter(~pl.col("pair").is_in(drop))
    return df


def run(variant: str = "main"):
    df = dataset(variant)
    styles = player_styles(df)
    feats = W.MODELS["B"][0]
    X = df.select(feats).to_numpy().astype(float)
    y, groups = df["y"].to_numpy(), df["pair"].to_numpy()
    rows, thr_rows = [], []
    for r in range(W.N_REPEATS):
        for k, (train, test) in enumerate(GroupKFold(W.N_SPLITS, shuffle=True, random_state=r).split(X, y, groups)):
            assert not set(groups[train]) & set(groups[test])
            s = W.fit_predict(X, y, train, test, [])
            inner = inner_scores(df, X, y, groups, train, seed=1000 + r).join(styles, on="player")
            clean = inner.filter(pl.col("y") == 0)
            thr_g = float(np.quantile(clean["score"].to_numpy(), 1 - TARGET_FPR))   # G0 (= wallhack_eval)
            thr_s = {}
            for st in STYLES:
                c = clean.filter(pl.col("stil") == st)["score"].to_numpy()
                fallback = len(c) < MIN_CLEAN
                thr_s[st] = thr_g if fallback else float(np.quantile(c, 1 - TARGET_FPR))
                thr_rows.append(dict(repeat=r, fold=k, stil=st, n_clean_train=len(c), fallback=fallback,
                                     thr=thr_s[st], thr_global=thr_g))
            pt = W.player_scores(df[test], s).join(styles, on="player")
            pt = pt.with_columns(repeat=pl.lit(r), flag_g0=pl.col("score") > thr_g,
                                 flag_g1=pl.col("score") > pl.col("stil").replace_strict(thr_s))
            rows.append(pt)
        print(f"tekrar {r} tamam", flush=True)
    suffix = "" if variant == "main" else f"_{variant}"
    pl.concat(rows).write_parquet(f"data/processed/style_flags{suffix}.parquet")
    pl.DataFrame(thr_rows).write_parquet(f"data/processed/style_thresholds{suffix}.parquet")
    print("kaydedildi")


def boot_idx(pairs_arr, seed):
    rng = np.random.default_rng(seed)
    uniq = np.unique(pairs_arr)
    idx = {q: np.where(pairs_arr == q)[0] for q in uniq}
    for _ in range(N_BOOT):
        yield np.concatenate([idx[q] for q in rng.choice(uniq, len(uniq))])


def report(variant: str = "main"):
    suffix = "" if variant == "main" else f"_{variant}"
    f = pl.read_parquet(f"data/processed/style_flags{suffix}.parquet")
    thr = pl.read_parquet(f"data/processed/style_thresholds{suffix}.parquet")
    p = f.group_by("player").agg(pl.col("y", "pair", "match", "stil").first(), score=pl.col("score").mean(),
                                 g0=pl.col("flag_g0").cast(pl.Float64).mean(), g1=pl.col("flag_g1").cast(pl.Float64).mean())
    print(f"#### {variant}: {p['pair'].n_unique()} çift, {p.height} oyuncu (temiz {(p['y'] == 0).sum()}, hileci {(p['y'] == 1).sum()})")

    # Kontrol: G0 tekrar başına metrikleri wallhack_eval ile aynı olmalı
    per_rep = f.group_by("repeat").agg(
        fpr0=pl.col("flag_g0").filter(pl.col("y") == 0).mean(), rec0=pl.col("flag_g0").filter(pl.col("y") == 1).mean(),
        fpr1=pl.col("flag_g1").filter(pl.col("y") == 0).mean(), rec1=pl.col("flag_g1").filter(pl.col("y") == 1).mean())
    print("Tekrar ortalaması: G0 FPR %.3f recall %.3f | G1 FPR %.3f recall %.3f" %
          tuple(per_rep[c].mean() for c in ["fpr0", "rec0", "fpr1", "rec1"]))

    # AUC: genel ve stil içi (eşikten bağımsız)
    print("\n== AUC (oyuncu, tekrar-ortalama skor; çift-bootstrap %95 GA) ==")
    for name, sub in [("genel", p)] + [(st, p.filter(pl.col("stil") == st)) for st in STYLES]:
        y, s, pr = sub["y"].to_numpy(), sub["score"].to_numpy(), sub["pair"].to_numpy()
        boots = [safe_auc(y[i], s[i]) for i in boot_idx(pr, 0)]
        lo, hi = np.nanpercentile(boots, [2.5, 97.5])
        print(f"  {name:20s} AUC {safe_auc(y, s):.3f} [{lo:.3f}–{hi:.3f}]  (temiz {(y == 0).sum()}, hileci {(y == 1).sum()})")

    # Karar metrikleri: nokta tahmin = oyuncu işaretlenme oranlarının ortalaması; eşleştirilmiş çift-bootstrap
    def metrics(sub, col):
        yy, ff = sub["y"].to_numpy(), sub[col].to_numpy()
        tp, fp = ff[yy == 1].sum(), ff[yy == 0].sum()
        fn, tn = (1 - ff[yy == 1]).sum(), (1 - ff[yy == 0]).sum()
        prec = tp / (tp + fp) if tp + fp else np.nan
        rec = tp / (tp + fn) if tp + fn else np.nan
        return dict(FPR=fp / (fp + tn) if fp + tn else np.nan, recall=rec, precision=prec,
                    F1=2 * prec * rec / (prec + rec) if prec and rec else np.nan, TP=tp, FP=fp, FN=fn, TN=tn)

    groups_ = [("genel", None)] + [(st, st) for st in STYLES]
    pr = p["pair"].to_numpy()
    boots = list(boot_idx(pr, 1))
    print("\n== Karar metrikleri: G0 → G1 [G1 − G0 eşleştirilmiş %95 GA] ==")
    for name, st in groups_:
        sub_mask = np.ones(p.height, bool) if st is None else (p["stil"] == st).to_numpy()
        base = p.filter(pl.Series(sub_mask))
        m0, m1 = metrics(base, "g0"), metrics(base, "g1")
        n_c, n_h = int((base["y"] == 0).sum()), int((base["y"] == 1).sum())
        keys = ["FPR", "recall", "precision", "F1"] if st is None else ["FPR", "recall"]
        cis = {k: [] for k in keys}
        ci0 = {k: [] for k in keys}
        ci1 = {k: [] for k in keys}
        for i in boots:
            b = p[i].filter(pl.Series(sub_mask[i]))
            a0, a1 = metrics(b, "g0"), metrics(b, "g1")
            for k in keys:
                cis[k].append(a1[k] - a0[k]); ci0[k].append(a0[k]); ci1[k].append(a1[k])
        print(f"  [{name}] temiz n={n_c}, hileci n={n_h}" + ("  (otomatik ağırlıklı hileci n küçük!)" if st == STYLES[1] else ""))
        for k in keys:
            lo, hi = np.nanpercentile(cis[k], [2.5, 97.5])
            l0, h0 = np.nanpercentile(ci0[k], [2.5, 97.5])
            l1, h1 = np.nanpercentile(ci1[k], [2.5, 97.5])
            print(f"     {k:9s} G0 {m0[k]:.3f} [{l0:.3f}–{h0:.3f}] → G1 {m1[k]:.3f} [{l1:.3f}–{h1:.3f}]"
                  f" | fark {m1[k] - m0[k]:+.3f} [{lo:+.3f}, {hi:+.3f}]")
        if st is None:
            print(f"     confusion (tekrar ortalaması): G0 TP {m0['TP']:.1f} FP {m0['FP']:.1f} FN {m0['FN']:.1f} TN {m0['TN']:.1f}"
                  f" | G1 TP {m1['TP']:.1f} FP {m1['FP']:.1f} FN {m1['FN']:.1f} TN {m1['TN']:.1f}")

    print("\n== Eşik oynaklığı (50 kat×tekrar) ==")
    print(thr.group_by("stil").agg(pl.col("thr").mean().round(3).alias("ort"), pl.col("thr").min().round(3).alias("min"),
                                   pl.col("thr").max().round(3).alias("max"), pl.col("n_clean_train").mean().round(1).alias("egitim_temiz_n"),
                                   pl.col("fallback").sum().alias("yedek_kural"))
             .sort("stil").to_pandas().to_string(index=False))
    g = thr.unique(["repeat", "fold"])["thr_global"]
    print(f"  global (G0) eşik: ort {g.mean():.3f}, min {g.min():.3f}, max {g.max():.3f}")

    oto_tp = p.filter((pl.col("stil") == STYLES[1]) & (pl.col("y") == 1) & (pl.col("g1") >= 0.5))
    print(f"\n== Yoğunlaşma: G1'de otomatik ağırlıklı hileci doğru pozitifleri (≥%50 tekrar): {oto_tp.height} oyuncu, "
          f"{oto_tp['match'].n_unique()} farklı maç ==")


if __name__ == "__main__":
    {"run": run, "report": report}[sys.argv[1]](*sys.argv[2:])
