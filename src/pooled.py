"""Havuzlu 160 çift değerlendirmesi (KEŞİFSEL; bağımsız doğrulama değildir).
Yöntem özeti: MODEL_CARD.md. Protokol wallhack_eval ile aynı (çift bazlı 5 kat × 10 tekrar, iç CV eşiği).
    python src/pooled.py ab          → A/B, B − A, stil içi AUC (çift-bootstrap GA)
    python src/pooled.py unmatched   → eşleşmeyen hileli maçların betimsel karşılaştırması (katalog)
"""
import sys

import numpy as np
import polars as pl

import wallhack_eval as W
from baseline import safe_auc
from select_matched import PER_MAP
from style_threshold import STYLES, player_styles

PAIRS_ALL = ("data/pairs.csv", "data/pairs_new.csv")
N_BOOT = 2000


def boot_auc(p, seed=0):
    rng = np.random.default_rng(seed)
    pr = p["pair"].to_numpy(); uniq = np.unique(pr)
    idx = {q: np.where(pr == q)[0] for q in uniq}
    y, s = p["y"].to_numpy(), p["score"].to_numpy()
    b = [safe_auc(y[i], s[i]) for i in
         (np.concatenate([idx[q] for q in rng.choice(uniq, len(uniq))]) for _ in range(N_BOOT))]
    return safe_auc(y, s), *np.nanpercentile(b, [2.5, 97.5])


def ab():
    from audit import team_results

    df = W.dataset(pairs_files=PAIRS_ALL)
    results = team_results(df["match"].unique().to_list())
    styles = player_styles(df)
    print(f"#### HAVUZLU (keşifsel): {df['pair'].n_unique()} çift, {df['match'].n_unique()} maç, {df.height} öldürme")
    out = {}
    for key in ["A", "B"]:
        feats, imp, label = W.MODELS[key]
        r = W.evaluate(df, feats, imp, results)
        out[key] = r
        p = r["players"].join(styles, on="player")
        print(f"\n{key} {label}")
        print(f"  kat-ortalama AUC {r['auc']:.3f} ± {r['auc_std']:.3f} | OOF AUC [GA] "
              f"{boot_auc(p)[0]:.3f} [{boot_auc(p)[1]:.3f}–{boot_auc(p)[2]:.3f}]")
        print(f"  G0 (iç CV, FPR %5): recall {r['recall'][0]:.3f} | FPR {r['fpr'][0]:.3f} | precision {r['precision'][0]:.3f}"
              f" | F1 {r['f1'][0]:.3f} | temiz-kazanan FPR {r['fpr_temiz_kazanan'][0]:.3f}")
        for st in STYLES:
            sp = p.filter(pl.col("stil") == st)
            a = boot_auc(sp)
            print(f"  [{st:18s}] AUC {a[0]:.3f} [{a[1]:.3f}–{a[2]:.3f}] | temiz {(sp['y'] == 0).sum()}, hileci {(sp['y'] == 1).sum()}")
    d, ci = W.paired_diff(out["A"]["players"], out["B"]["players"])
    print(f"\nB − A (OOF, eşleştirilmiş çift-bootstrap): {d:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}]")
    pa, pb = out["A"]["players"].join(styles, on="player"), out["B"]["players"].join(styles, on="player")
    for st in STYLES:
        d, ci = W.paired_diff(pa.filter(pl.col("stil") == st), pb.filter(pl.col("stil") == st))
        print(f"  B − A [{st}]: {d:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}]")


def unmatched():
    cat = pl.read_csv("data/catalog.csv").with_columns(premier=pl.col("match_making_type").str.contains("Premier"))
    pairs = pl.concat([pl.read_csv(f) for f in PAIRS_ALL])
    matched = set(pairs["cheat"].to_list())
    ch = cat.filter(pl.col("folder") == "with_cheater_present").with_columns(
        durum=pl.when(pl.col("match").is_in(list(matched))).then(pl.lit("eşleşen"))
               .when(pl.col("map").is_in(list(PER_MAP))).then(pl.lit("eşleşmeyen (9 harita)"))
               .otherwise(pl.lit("eşleşmeyen (diğer harita)")))
    print("#### Hileli maçlar: eşleşen vs eşleşmeyen (BETİMSEL; test sonuçlarına karışmaz)")
    print(ch.group_by("durum").agg(pl.len().alias("maç"), pl.col("size_mb").median().alias("boyut_medyan_mb"),
                                   pl.col("size_mb").quantile(0.25).alias("q25"), pl.col("size_mb").quantile(0.75).alias("q75"),
                                   pl.col("premier").mean().round(2).alias("premier_oranı"),
                                   pl.col("n_cheaters").mean().round(2).alias("ort_hileci"))
            .sort("durum").to_pandas().to_string(index=False))
    m = ch.group_by("durum", "map").len().pivot(on="durum", index="map", values="len").fill_null(0).sort("map")
    print("\nharita dağılımı:\n" + m.to_pandas().to_string(index=False))
    # Neden eşleşmedi? Aynı harita+mod'da boyutça ±%25 temiz maç var mıydı (kullanılmış olsun olmasın)?
    clean = cat.filter(pl.col("folder") == "no_cheater_present")
    um = ch.filter(pl.col("durum") == "eşleşmeyen (9 harita)")
    def has_cand(r):
        c = clean.filter((pl.col("map") == r["map"]) & (pl.col("premier") == r["premier"]))["size_mb"].to_numpy()
        if not len(c):
            return False
        ratio = np.maximum(c, r["size_mb"]) / np.minimum(c, r["size_mb"])
        return bool((ratio <= 1.25).any())
    um = um.with_columns(aday_vardı=pl.Series([has_cand(r) for r in um.iter_rows(named=True)]))
    print("\neşleşmeyen (9 harita): uygun boyutta temiz aday hiç yoktu:", (~um["aday_vardı"]).sum(),
          "| aday vardı ama hepsi kullanılmıştı:", um["aday_vardı"].sum())
    print("rütbe: avg_rank alanı katalogda metin/puan karışık; Premier dışı maçlarda rütbe dağılımı:")
    rk = ch.filter(~pl.col("premier")).group_by("durum", "avg_rank").len().sort("durum", "len", descending=[False, True])
    for d in rk["durum"].unique().sort().to_list():
        print(f"  {d}: " + ", ".join(f"{a} {n}" for a, n in rk.filter(pl.col("durum") == d).select("avg_rank", "len").head(6).iter_rows()))


if __name__ == "__main__":
    {"ab": ab, "unmatched": unmatched}[sys.argv[1]]()
