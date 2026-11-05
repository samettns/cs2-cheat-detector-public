"""Duyarlılık analizleri S1–S3 (yöntem özeti: MODEL_CARD.md).

Ana analiz değişmez; hiçbir oyuncu yeniden etiketlenmez. Rapor: MODEL_CARD.md.
    python src/sensitivity.py s1s2     # S1 (3 maç hariç) ve S2 (modelden bağımsız kural) — A ve B yeniden değerlendirilir
    python src/sensitivity.py s3       # silah stili hata analizi (ana analizin karar kayıtları)
"""
import sys

import numpy as np
import polars as pl
from scipy.stats import fisher_exact

import wallhack_eval as W
from features import WEAPON_CLASS
from load import RAW_DIR, event_df, load_events

S1_MATCHES = ["no_cheater_present/254", "no_cheater_present/246", "no_cheater_present/68"]
SNIPER_SHARE_MAX = 0.5      # S2: maç genelinde sniper öldürme payı bu değerden büyükse standart dışı
STYLE_SHARE = 0.5           # S3: oyuncu stili eşiği (öldürmelerin > %50'si)
N_BOOT = 2000


def match_sniper_share(match: str) -> float:
    d = event_df(load_events(RAW_DIR / match), "player_death").filter(
        pl.col("attacker_steamid").is_not_null() & (pl.col("attacker_steamid") != pl.col("user_steamid")))
    return d.select(pl.col("weapon").replace_strict(WEAPON_CLASS, default="diger") == "sniper").mean().item()


def evaluate_pairs(df: pl.DataFrame, keep_pairs, label: str):
    from audit import team_results

    sub = df.filter(pl.col("pair").is_in(list(keep_pairs)))
    results = team_results(sub["match"].unique().to_list())
    print(f"\n######## {label}: {sub['pair'].n_unique()} çift, {sub['match'].n_unique()} maç, "
          f"{sub.height} öldürme ########", flush=True)
    out = {}
    for key in ["A", "B"]:
        feats, imp, name = W.MODELS[key]
        r = W.evaluate(sub, feats, imp, results)
        out[key] = r
        print(f"{key} {name}\n   AUC {r['auc']:.3f} ± {r['auc_std']:.3f} [{r['auc_ci'][0]:.2f}–{r['auc_ci'][1]:.2f}]"
              f" | recall@FPR%5 {r['recall'][0]:.3f}±{r['recall'][1]:.3f} | FPR {r['fpr'][0]:.3f}"
              f" | FPR temiz-kazanan {r['fpr_temiz_kazanan'][0]:.3f}, temiz-kaybeden {r['fpr_temiz_kaybeden'][0]:.3f}",
              flush=True)
    d, ci = W.paired_diff(out["A"]["players"], out["B"]["players"])
    print(f"   B − A: {d:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}]", flush=True)


def s1s2():
    df = W.dataset(False)
    pairs = pl.read_csv("data/pairs.csv")
    all_pairs = set(pairs["pair"].to_list())

    # S1: 3 maç (ve çiftleri) hariç — döngüsel seçim, sadece "en kötü durum" duyarlılığı
    drop1 = set(pairs.filter(pl.col("clean").is_in(S1_MATCHES) | pl.col("cheat").is_in(S1_MATCHES))["pair"].to_list())
    evaluate_pairs(df, all_pairs - drop1, f"S1: maç 254/246/68 ve çiftleri hariç (çıkan çiftler {sorted(drop1)})")

    # S2: modelden bağımsız kural, 100 maçın hepsine
    rows = []
    for p in pairs.iter_rows(named=True):
        for side in ("clean", "cheat"):
            rows.append(dict(pair=p["pair"], match=p[side], grup="temiz" if side == "clean" else "hileli",
                             map=p["map"], sniper_share=match_sniper_share(p[side])))
    share = pl.DataFrame(rows).with_columns(standart_disi=pl.col("sniper_share") > SNIPER_SHARE_MAX)
    share.write_csv("data/processed/s2_match_sniper_share.csv")
    flagged = share.filter(pl.col("standart_disi"))
    print("\n== S2 kuralı: maç geneli sniper öldürme payı > %50 ==")
    print(flagged.select("match", "grup", "map", pl.col("sniper_share").round(2)).sort("grup", "match")
                 .to_pandas().to_string(index=False))
    print("standart dışı maç sayısı:", flagged.group_by("grup").len().sort("grup").to_dicts(),
          "| dağılım (tüm 100 maç) medyan/q90:", round(share["sniper_share"].median(), 2),
          round(share["sniper_share"].quantile(0.9), 2))
    drop2 = set(flagged["pair"].to_list())
    print("S1 maçları kurala takılıyor mu:", flagged.filter(pl.col("match").is_in(S1_MATCHES))["match"].to_list())
    evaluate_pairs(df, all_pairs - drop2, f"S2: kurala takılan maçların çiftleri hariç (çıkan çiftler {len(drop2)})")


def boot_mean(p: pl.DataFrame, col: str, seed=0):
    rng = np.random.default_rng(seed)
    pairs = p["pair"].unique().to_numpy()
    idx = {q: np.where(p["pair"].to_numpy() == q)[0] for q in pairs}
    v = p[col].to_numpy()
    est = [v[np.concatenate([idx[q] for q in rng.choice(pairs, len(pairs))])].mean() for _ in range(N_BOOT)]
    return v.mean(), *np.percentile(est, [2.5, 97.5])


def s3():
    from fp_analysis import FP_RULE, per_player

    df = W.dataset(False)
    style = df.group_by("player").agg(sniper=pl.col("is_sniper").mean(),
                                      oto=(pl.col("weapon_class") == "otomatik").mean())
    style = style.with_columns(stil=pl.when(pl.col("sniper") > STYLE_SHARE).then(pl.lit("sniper ağırlıklı"))
                                      .when(pl.col("oto") > STYLE_SHARE).then(pl.lit("otomatik ağırlıklı"))
                                      .otherwise(pl.lit("diğer")))
    for model in ["A", "B"]:
        p = per_player(model).join(style.select("player", "stil"), on="player")
        print(f"\n######## Model {model} ########")
        rows = []
        for st in ["sniper ağırlıklı", "otomatik ağırlıklı", "diğer"]:
            c = p.filter((pl.col("stil") == st) & (pl.col("y") == 0))
            h = p.filter((pl.col("stil") == st) & (pl.col("y") == 1))
            fpr = boot_mean(c, "flag_rate") if c.height else (np.nan,) * 3
            rec = boot_mean(h, "flag_rate") if h.height else (np.nan,) * 3
            rows.append(dict(stil=st, temiz_n=c.height, FPR=round(fpr[0], 3), FPR_GA=f"[{fpr[1]:.3f}–{fpr[2]:.3f}]",
                             FP_oyuncu=c.filter(pl.col("flag_rate") >= FP_RULE).height,
                             hileci_n=h.height, recall=round(rec[0], 3), recall_GA=f"[{rec[1]:.3f}–{rec[2]:.3f}]"))
        print(pl.DataFrame(rows).to_pandas().to_string(index=False))

        # Sniper ağırlıklı temizlerde FPR farkı: sniper − diğer stiller (çift-bootstrap)
        clean = p.filter(pl.col("y") == 0).with_columns(is_sn=pl.col("stil") == "sniper ağırlıklı")
        rng = np.random.default_rng(1)
        pairs = clean["pair"].unique().to_numpy()
        idx = {q: np.where(clean["pair"].to_numpy() == q)[0] for q in pairs}
        fr, sn = clean["flag_rate"].to_numpy(), clean["is_sn"].to_numpy()
        diffs = []
        for _ in range(N_BOOT):
            i = np.concatenate([idx[q] for q in rng.choice(pairs, len(pairs))])
            if sn[i].any() and (~sn[i]).any():
                diffs.append(fr[i][sn[i]].mean() - fr[i][~sn[i]].mean())
        d = fr[sn].mean() - fr[~sn].mean()
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        fp = clean.with_columns(fp=pl.col("flag_rate") >= FP_RULE)
        tab = [[fp.filter(pl.col("is_sn") & pl.col("fp")).height, fp.filter(pl.col("is_sn") & ~pl.col("fp")).height],
               [fp.filter(~pl.col("is_sn") & pl.col("fp")).height, fp.filter(~pl.col("is_sn") & ~pl.col("fp")).height]]
        odds, pv = fisher_exact(tab)
        print(f"FPR farkı (sniper ağırlıklı − diğer stiller): {d:+.3f} [{lo:+.3f}, {hi:+.3f}] | "
              f"FP tablosu {tab} Fisher p={pv:.4f}, odds={odds:.1f}")
        # S1 maçları hariç aynı fark (keşif)
        c2 = clean.filter(~pl.col("match").is_in(S1_MATCHES))
        print(f"   (keşif) 254/246/68 hariç: sniper FPR {c2.filter(pl.col('is_sn'))['flag_rate'].mean():.3f} "
              f"(n={c2.filter(pl.col('is_sn')).height}) vs diğer {c2.filter(~pl.col('is_sn'))['flag_rate'].mean():.3f}")


if __name__ == "__main__":
    {"s1s2": s1s2, "s3": s3}[sys.argv[1]]()
