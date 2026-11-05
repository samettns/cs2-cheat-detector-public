"""Aşama 2: wallhack-benzeri özelliklerin model katkısı ve NaN duyarlılık analizi.

Yöntem özeti: MODEL_CARD.md. Rapor: MODEL_CARD.md
    python src/wallhack_eval.py run        # 50 çift ve 45 çift (anubis'siz), tüm modeller
    python src/wallhack_eval.py extras     # NaN nedenleri, korelasyon, permutation importance
    python src/wallhack_eval.py ab         # yeniden üretim: sadece ana A/B (50 çift)
"""
import sys

import numpy as np
import polars as pl
from sklearn.model_selection import GroupKFold

from baseline import AIM_FEATURES, VISIBILITY_FEATURES, WEAPON_FEATURES, aggregate, load, make_model, safe_auc

N_SPLITS, N_REPEATS, INNER_SPLITS = 5, 10, 4
TARGET_FPR = 0.05
MIN_KILLS = 3
N_BOOT = 1000
VIS = VISIBILITY_FEATURES   # tek kaynak: baseline.py
MISSING = ["prefov_missing", "reaction_missing", "combined_missing"]
IMPUTE = ["prefov_angle", "reaction_ms"]
BASE = AIM_FEATURES + WEAPON_FEATURES

# ad: (özellikler, eğitim-katı medyanıyla doldurulacak sütunlar, etiket)
MODELS = {
    "A":   (BASE, [], "baseline (nişan + silah)"),
    "B":   (BASE + VIS, [], "baseline + görünürlük, native NaN  [ANA]"),
    "C":   (BASE + VIS, IMPUTE, "baseline + görünürlük, eğitim medyanı ile doldurma  [duyarlılık]"),
    "D":   (MISSING, [], "sadece eksiklik göstergeleri  [kontrol]"),
    "W":   (["wallbang"], [], "sadece wallbang  [kontrol]"),
    "D+W": (MISSING + ["wallbang"], [], "eksiklik + wallbang  [kontrol]"),
    "V":   (VIS, [], "sadece görünürlük grubu  [betimleyici]"),
}


def dataset(drop_anubis: bool = False, pairs_files=("data/pairs.csv",)) -> pl.DataFrame:
    """pairs_files: varsayılan eski 50 çift; büyütülmüş veri için ("data/pairs.csv", "data/pairs_new.csv")."""
    pairs = pl.concat([pl.read_csv(f) for f in pairs_files])
    if drop_anubis:
        pairs = pairs.filter(pl.col("map") != "de_anubis")
    pair_of = pl.concat([pairs.select("pair", match="cheat"), pairs.select("pair", match="clean")])
    # 5 görünürlük sütunu artık features.py hattından (kill_features.parquet) geliyor; ayrı dosyadan
    # sadece teşhis sütunları (NaN nedenleri vb.) alınır.
    wf = (pl.read_parquet("data/processed/wallhack_features.parquet").rename({"kill_tick": "tick"})
            .drop(VISIBILITY_FEATURES))
    df = (load().join(pair_of, on="match").join(wf, on=["match", "player", "tick"], how="left")
                .sort("match", "player", "tick"))
    return df.with_columns(
        prefov_missing=pl.col("prefov_angle").is_null().cast(pl.Int8),
        reaction_missing=pl.col("reaction_ms").is_null().cast(pl.Int8),
        combined_missing=(pl.col("prefov_angle").is_null() | pl.col("reaction_ms").is_null()).cast(pl.Int8),
    )


def impute_with(X_fit, X_apply, cols_idx):
    """cols_idx sütunlarındaki NaN'ları X_fit'in (eğitim) medyanıyla doldurur; kopyalar döner."""
    X_fit, X_apply = X_fit.copy(), X_apply.copy()
    for j in cols_idx:
        med = np.nanmedian(X_fit[:, j])
        X_fit[np.isnan(X_fit[:, j]), j] = med
        X_apply[np.isnan(X_apply[:, j]), j] = med
    return X_fit, X_apply


def player_scores(df_sub, kill_scores):
    return aggregate(df_sub, kill_scores, "mean", MIN_KILLS, group_col="pair")


def fit_predict(X, y, train, test, imp_idx):
    Xtr, Xte = impute_with(X[train], X[test], imp_idx)
    return make_model().fit(Xtr, y[train]).predict_proba(Xte)[:, 1]


def inner_threshold(df, X, y, groups, train, imp_idx, seed):
    """Eğitim katı içinde iç CV ile dışarıda-kalan skorlar → temizlerde FPR %5 eşiği."""
    inner = np.zeros(len(train))
    Xt, yt, gt = X[train], y[train], groups[train]
    for itr, ite in GroupKFold(INNER_SPLITS, shuffle=True, random_state=seed).split(Xt, yt, gt):
        inner[ite] = fit_predict(Xt, yt, itr, ite, imp_idx)
    p = player_scores(df[train], inner)
    clean = p.filter(pl.col("y") == 0)["score"].to_numpy()
    return float(np.quantile(clean, 1 - TARGET_FPR))


def threshold_metrics(p, results=None):
    """p: oyuncu tablosu (y, score, flag). FPR, recall, precision, F1 (+ kazanan/kaybeden temiz FPR)."""
    y, f = p["y"].to_numpy(), p["flag"].to_numpy()
    tp, fp = int(((y == 1) & f).sum()), int(((y == 0) & f).sum())
    fn, tn = int(((y == 1) & ~f).sum()), int(((y == 0) & ~f).sum())
    rec = tp / (tp + fn) if tp + fn else np.nan
    prec = tp / (tp + fp) if tp + fp else np.nan
    out = dict(fpr=fp / (fp + tn), recall=rec, precision=prec,
               f1=2 * prec * rec / (prec + rec) if prec and rec else 0.0, tp=tp, fp=fp, fn=fn, tn=tn)
    if results is not None:
        q = p.join(results, on="player", how="left")
        for res in ["kazanan", "kaybeden"]:
            c = q.filter((pl.col("y") == 0) & (pl.col("result") == res))
            out[f"fpr_temiz_{res}"] = c["flag"].mean() if c.height else np.nan
    return out


def evaluate(df, feats, imp_cols, results, n_repeats=N_REPEATS):
    X = df.select(feats).to_numpy().astype(float)
    y = df["y"].to_numpy()
    groups = df["pair"].to_numpy()
    imp_idx = [feats.index(c) for c in imp_cols]
    fold_aucs, rep_metrics, oof_sum = [], [], np.zeros(len(y))
    for r in range(n_repeats):
        oof = np.zeros(len(y))
        flags = []
        for train, test in GroupKFold(N_SPLITS, shuffle=True, random_state=r).split(X, y, groups):
            assert not set(groups[train]) & set(groups[test])   # aynı çift/maç iki tarafta olamaz
            oof[test] = fit_predict(X, y, train, test, imp_idx)
            thr = inner_threshold(df, X, y, groups, train, imp_idx, seed=1000 + r)
            pt = player_scores(df[test], oof[test])
            fold_aucs.append(safe_auc(pt["y"], pt["score"]))
            flags.append(pt.with_columns(flag=pl.col("score") > thr))
        rep_metrics.append(threshold_metrics(pl.concat(flags), results))
        oof_sum += oof
    players = player_scores(df, oof_sum / n_repeats)
    m = pl.DataFrame(rep_metrics)
    return dict(auc=np.nanmean(fold_aucs), auc_std=np.nanstd(fold_aucs),
                auc_ci=boot_ci(players), players=players,
                **{k: (m[k].mean(), m[k].std()) for k in m.columns})


def boot_ci(players, seed=0):
    rng = np.random.default_rng(seed)
    pairs = players["pair"].unique().to_numpy()
    idx = {p: np.where(players["pair"].to_numpy() == p)[0] for p in pairs}
    y, s = players["y"].to_numpy(), players["score"].to_numpy()
    aucs = [safe_auc(y[i], s[i]) for i in
            (np.concatenate([idx[p] for p in rng.choice(pairs, len(pairs))]) for _ in range(N_BOOT))]
    return tuple(np.nanpercentile(aucs, [2.5, 97.5]))


def paired_diff(pa, pb, seed=0):
    """Aynı oyuncular, aynı yeniden örneklenmiş çiftler: AUC_b − AUC_a ve %95 GA."""
    j = pa.select("player", "pair", "y", sa="score").join(pb.select("player", sb="score"), on="player")
    rng = np.random.default_rng(seed)
    pairs = j["pair"].unique().to_numpy()
    idx = {p: np.where(j["pair"].to_numpy() == p)[0] for p in pairs}
    y, sa, sb = j["y"].to_numpy(), j["sa"].to_numpy(), j["sb"].to_numpy()
    diffs = []
    for _ in range(N_BOOT):
        i = np.concatenate([idx[p] for p in rng.choice(pairs, len(pairs))])
        diffs.append(safe_auc(y[i], sb[i]) - safe_auc(y[i], sa[i]))
    return safe_auc(y, sb) - safe_auc(y, sa), tuple(np.nanpercentile(diffs, [2.5, 97.5]))


def run():
    from audit import team_results

    for name, drop in [("50 eşleştirilmiş çift (anubis maskeli)", False), ("45 çift, anubis'siz [duyarlılık]", True)]:
        df = dataset(drop)
        results = team_results(df["match"].unique().to_list())
        print(f"\n######## {name}: {df['pair'].n_unique()} çift, {df['match'].n_unique()} maç, "
              f"{df.height} öldürme, {df['player'].n_unique()} oyuncu ########")
        out = {}
        for key, (feats, imp, label) in MODELS.items():
            r = evaluate(df, feats, imp, results)
            out[key] = r
            print(f"{key:4s} {label}\n     AUC {r['auc']:.3f} ± {r['auc_std']:.3f} [{r['auc_ci'][0]:.2f}–{r['auc_ci'][1]:.2f}]"
                  f" | FPR%5 eşiği: recall {r['recall'][0]:.3f}±{r['recall'][1]:.3f}  FPR {r['fpr'][0]:.3f}"
                  f"  precision {r['precision'][0]:.3f}  F1 {r['f1'][0]:.3f}"
                  f" | FPR temiz-kazanan {r['fpr_temiz_kazanan'][0]:.3f}, temiz-kaybeden {r['fpr_temiz_kaybeden'][0]:.3f}",
                  flush=True)
        print("\n  Eşleştirilmiş çift-bootstrap AUC farkları (aynı katlar, OOF oyuncu skorları):")
        for a, b in [("A", "B"), ("A", "C"), ("B", "C"), ("W", "D+W"), ("D", "D+W"), ("A", "V")]:
            d, ci = paired_diff(out[a]["players"], out[b]["players"])
            print(f"   {b} − {a}: {d:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}]")
        # hileci vs kazanan/kaybeden temiz AUC (OOF ortalama skorlarla)
        print("\n  Hileci vs temiz-kazanan / temiz-kaybeden AUC:")
        for key in ["A", "B", "C"]:
            q = out[key]["players"].join(results, on="player", how="left")
            ch = q.filter(pl.col("y") == 1)
            vals = []
            for res in ["kazanan", "kaybeden"]:
                c = pl.concat([ch, q.filter((pl.col("y") == 0) & (pl.col("result") == res))])
                vals.append(safe_auc(c["y"], c["score"]))
            print(f"   {key}: vs kazanan {vals[0]:.3f} | vs kaybeden {vals[1]:.3f}")


def ab():
    """Yeniden üretim kontrolü: ana A/B karşılaştırması (50 çift), `run` ile aynı protokol ve çıktı biçimi."""
    from audit import team_results

    df = dataset(False)
    results = team_results(df["match"].unique().to_list())
    print(f"######## 50 eşleştirilmiş çift (anubis maskeli): {df['pair'].n_unique()} çift, {df['match'].n_unique()} maç, "
          f"{df.height} öldürme, {df['player'].n_unique()} oyuncu ########")
    out = {}
    for key in ["A", "B"]:
        feats, imp, label = MODELS[key]
        r = evaluate(df, feats, imp, results)
        out[key] = r
        print(f"{key:4s} {label}\n     AUC {r['auc']:.3f} ± {r['auc_std']:.3f} [{r['auc_ci'][0]:.2f}–{r['auc_ci'][1]:.2f}]"
              f" | FPR%5 eşiği: recall {r['recall'][0]:.3f}±{r['recall'][1]:.3f}  FPR {r['fpr'][0]:.3f}"
              f"  precision {r['precision'][0]:.3f}  F1 {r['f1'][0]:.3f}"
              f" | FPR temiz-kazanan {r['fpr_temiz_kazanan'][0]:.3f}, temiz-kaybeden {r['fpr_temiz_kaybeden'][0]:.3f}",
              flush=True)
        print(f"     confusion (10 tekrar ortalaması, oyuncu): TP {r['tp'][0]:.1f}  FP {r['fp'][0]:.1f}  "
              f"FN {r['fn'][0]:.1f}  TN {r['tn'][0]:.1f}", flush=True)
    d, ci = paired_diff(out["A"]["players"], out["B"]["players"])
    print(f"   B − A: {d:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}]")


def extras():
    from scipy.stats import spearmanr

    df = dataset(False)
    # 1) NaN nedenleri (pre-FOV / tepki süresi)
    reason = (pl.when(pl.col("f2_nan") == "duman").then(pl.lit("duman"))
                .when(pl.col("f2_nan") == "maske").then(pl.lit("anubis maskesi"))
                .when((pl.col("f2_nan") == "gecis_yok") & (pl.col("wallbang") == 1)).then(pl.lit("duvar delme (geçiş yok)"))
                .when(pl.col("f2_nan") == "gecis_yok").then(pl.lit("geçiş yok, duvar delmeden"))
                .when(pl.col("f2_nan") == "pencere_disi").then(pl.lit("pencere dışı"))
                .when(pl.col("reaction_ms").is_null()).then(pl.lit("hasar kaydı yok (sadece tepki)"))
                .otherwise(pl.lit("eksik değil")))
    t = df.with_columns(neden=reason).group_by("group", "neden").len()
    t = t.with_columns(oran=(pl.col("len") / pl.col("len").sum().over("group") * 100).round(1))
    print("== NaN nedenleri (pre-FOV/tepki süresi), öldürme sayısı ve grup içi % ==")
    print(t.pivot(on="group", index="neden", values=["len", "oran"]).sort("neden").to_pandas().to_string(index=False))

    # 2) Spearman korelasyonu
    cols = AIM_FEATURES + VIS
    m = df.select(cols).to_pandas()
    rho = spearmanr(m, nan_policy="omit").statistic
    print("\n== Spearman korelasyonu (|rho| ≥ 0.3 olan çiftler) ==")
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            if abs(rho[i, j]) >= 0.3:
                print(f"   {cols[i]:18s} ~ {cols[j]:18s} rho={rho[i, j]:+.2f}")

    # 3) Permutation importance (model B): grup ve tekil; oyuncu AUC düşüşü, test katlarında
    feats = BASE + VIS
    X = df.select(feats).to_numpy().astype(float)
    y, groups = df["y"].to_numpy(), df["pair"].to_numpy()
    groups_def = {"nişan grubu": AIM_FEATURES, "silah grubu": WEAPON_FEATURES, "görünürlük grubu": VIS}
    singles = {f: [f] for f in feats}
    rng = np.random.default_rng(0)
    drops = {k: [] for k in list(groups_def) + list(singles)}
    for r in range(3):
        for train, test in GroupKFold(N_SPLITS, shuffle=True, random_state=r).split(X, y, groups):
            model = make_model().fit(X[train], y[train])
            base = safe_auc(*player_scores(df[test], model.predict_proba(X[test])[:, 1]).select("y", "score"))
            for name, cs in {**groups_def, **singles}.items():
                idx = [feats.index(c) for c in cs]
                for _ in range(3):
                    Xp = X[test].copy()
                    perm = rng.permutation(len(test))
                    Xp[:, idx] = Xp[perm][:, idx]   # grup sütunları birlikte karıştırılır
                    s = model.predict_proba(Xp)[:, 1]
                    drops[name].append(base - safe_auc(*player_scores(df[test], s).select("y", "score")))
    print("\n== Permutation importance, model B (oyuncu AUC düşüşü, 3 tekrar × 5 kat × 3 karıştırma) ==")
    for name in list(groups_def) + sorted(singles, key=lambda k: -np.mean(drops[k])):
        print(f"   {name:20s} {np.mean(drops[name]):+.3f} ± {np.std(drops[name]):.3f}")
    print("   Not: ilişkili özelliklerde tekil önem yanıltıcı olabilir (biri karıştırılınca diğeri bilgiyi taşır);"
          " grup sonuçları birincil.")


if __name__ == "__main__":
    {"run": run, "extras": extras, "ab": ab}[sys.argv[1]]()
