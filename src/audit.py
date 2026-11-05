"""Baseline denetimi (audit): metadata, harita yanlılığı, oyuncu skoru birleştirme.

    python src/audit.py metadata
    python src/audit.py maps
    python src/audit.py aggregation
    python src/audit.py matched      # harita-eşleşmeli veri setinde ana deneyler
    python src/audit.py pairs        # harita + mod + uzunluk eşleşmeli çiftlerde ana deneyler
"""
import sys

import numpy as np
import polars as pl

from baseline import AIM_FEATURES, FEATURE_SETS, WEAPON_FEATURES, aggregate, evaluate, fmt, load, safe_auc
from load import RAW_DIR, load_events

MAIN = AIM_FEATURES + WEAPON_FEATURES  # şu an raporlanan ana model


def match_metadata() -> pl.DataFrame:
    """Her maç için davranış dışı bilgiler."""
    rows = []
    for line in open("data/manifest.txt", encoding="utf-8").read().split():
        ev = load_events(RAW_DIR / line)
        info = ev["CSstats_info"][0]
        cvar = {c["name"]: c["value"] for c in ev.get("server_cvar", [])}
        rank = info["avg_rank"]
        rows.append(dict(
            match=line,
            map=info["map"],
            server=info["server"],
            match_making_type=info["match_making_type"],
            # Premier maçlarda rütbe yerine sayısal puan yazıyor
            avg_rank="premier_puan" if rank.isdigit() else rank,
            maxrounds=cvar.get("mp_maxrounds", "?"),
            session_id=float(cvar.get("steamworks_sessionid_server", "nan")),  # zaman göstergesi olabilir
        ))
    return pl.DataFrame(rows)


def one_hot(df, cols):
    dummies = df.select(cols).to_dummies()
    return df.hstack(dummies), dummies.columns


def run_metadata():
    meta = match_metadata()
    df = load().join(meta, on="match").sort("match", "player", "tick")
    print("Maç düzeyi dağılım (temiz | hileli):")
    mt = meta.with_columns(grup=pl.col("match").str.split("/").list.first())
    for c in ["server", "match_making_type", "avg_rank", "maxrounds"]:
        t = mt.group_by(c, "grup").len().pivot(on="grup", index=c, values="len").fill_null(0)
        print(t.sort(c).to_pandas().to_string(index=False), "\n")
    s = mt.group_by("grup").agg(pl.col("session_id").min().alias("min"), pl.col("session_id").max().alias("max"))
    print("session_id aralığı:", s.to_dicts(), "\n")

    print(f"{df.height} öldürme, {df['match'].n_unique()} maç\n")
    for name, cols, numeric in [
        ("metadata (map+rank+server+tür)", ["map", "avg_rank", "server", "match_making_type"], []),
        ("sadece map", ["map"], []),
        ("sadece avg_rank", ["avg_rank"], []),
        ("sadece server", ["server"], []),
        ("sadece session_id (zaman?)", [], ["session_id"]),
    ]:
        d, feats = one_hot(df, cols) if cols else (df, [])
        print(f"{name:32s} {fmt(evaluate(d, feats + numeric))}")


def run_maps():
    meta = match_metadata()
    mt = meta.with_columns(grup=pl.col("match").str.split("/").list.first())
    table = (mt.group_by("map", "grup").len().pivot(on="grup", index="map", values="len").fill_null(0)
               .rename({"no_cheater_present": "temiz", "with_cheater_present": "hileli"}).sort("map"))
    print(table.to_pandas().to_string(index=False), "\n")

    # Dengeli alt küme: her haritadan min(temiz, hileli) kadar maç, sabit seed ile
    rng = np.random.default_rng(42)
    keep = []
    for m, t, h in table.select("map", "temiz", "hileli").iter_rows():
        k = min(t, h)
        for grup, n in [("no_cheater_present", t), ("with_cheater_present", h)]:
            cands = sorted(mt.filter((pl.col("map") == m) & (pl.col("grup") == grup))["match"].to_list())
            keep += list(rng.choice(cands, k, replace=False)) if k else []
    df = load()
    bal = df.filter(pl.col("match").is_in(keep))
    print(f"Dengeli alt küme: {len(keep)} maç ({len(keep) // 2} temiz + {len(keep) // 2} hileli), "
          f"{bal.height} öldürme, {bal['player'].n_unique()} oyuncu\n")
    for name, feats in [("tüm veri, nişan + silah", MAIN)]:
        print(f"{name:34s} {fmt(evaluate(df, feats))}")
    for name, feats in [("dengeli, nişan + silah", MAIN), ("dengeli, sadece silah", WEAPON_FEATURES),
                        ("dengeli, sadece nişan", AIM_FEATURES)]:
        r = evaluate(bal, feats)
        print(f"{name:34s} {fmt(r)} | {r['n_players']} oyuncu")


def run_aggregation():
    df = load()
    print(f"{df.height} öldürme, {df['match'].n_unique()} maç. Model: nişan + silah. Birincil yöntem: mean\n")
    oof = evaluate(df, MAIN, n_boot=100)["oof"]  # tekrarlar boyunca ortalanmış OOF skorlar (GA burada kullanılmıyor)
    print(f"{'yöntem':10s} " + " ".join(f"{'min ' + str(k) + ' öld.':>18s}" for k in [1, 3, 5]))
    for method in ["mean", "median", "max", "top3_mean"]:
        cells = []
        for k in [1, 3, 5]:
            p = aggregate(df, oof, method, min_kills=k)
            cells.append(f"{safe_auc(p['y'], p['score']):.3f} ({p.height} oy.)")
        print(f"{method:10s} " + " ".join(f"{c:>18s}" for c in cells))


def run_matched():
    """Harita-eşleşmeli veri setinde (data/matched_manifest.txt) ana denetim deneyleri."""
    matched = open("data/matched_manifest.txt", encoding="utf-8").read().split()
    catalog = pl.read_csv("data/catalog.csv").select("match", "map", "avg_rank", "server",
                                                     "match_making_type", "size_mb")
    df = load().filter(pl.col("match").is_in(matched)).join(catalog, on="match").sort("match", "player", "tick")
    n_t = df.filter(pl.col("y") == 0)["match"].n_unique()
    n_h = df.filter(pl.col("y") == 1)["match"].n_unique()
    print(f"Harita-eşleşmeli: {df['match'].n_unique()} maç ({n_t} temiz + {n_h} hileli), "
          f"{df.height} öldürme, {df['player'].n_unique()} oyuncu\n")

    print("== Özellik ablasyonu ==")
    for name, feats in FEATURE_SETS.items():
        r = evaluate(df, feats)
        print(f"{name:28s} {fmt(r)} | {r['n_players']} oy.")
    no_shots = [f for f in AIM_FEATURES if f != "shots"]
    print(f"{'nişan (shots hariç)':28s} {fmt(evaluate(df, no_shots))}")

    print("\n== Silah sınıfı içinde, sadece nişan ==")
    for wc in ["sniper", "tabanca", "otomatik"]:
        sub = df.filter(pl.col("weapon_class") == wc)
        r = evaluate(sub, AIM_FEATURES)
        print(f"{wc:10s} {fmt(r)} | {r['n_kills']} öld., {r['n_players']} oy., {r['n_matches']} maç")

    print("\n== Metadata (davranış dışı) ==")
    for name, cols, numeric in [
        ("sadece map (dengeli → ~0.5 beklenir)", ["map"], []),
        ("map + rank + server + tür", ["map", "avg_rank", "server", "match_making_type"], []),
        ("dosya boyutu (maç uzunluğu)", [], ["size_mb"]),
    ]:
        d, feats = one_hot(df, cols) if cols else (df, [])
        print(f"{name:38s} {fmt(evaluate(d, feats + numeric))}")


def match_stats(matches) -> pl.DataFrame:
    """Gerçek maç uzunluğu ve tek taraflılık: süre (dk), round sayısı, final skor farkı."""
    rows = []
    for m in matches:
        ev = load_events(RAW_DIR / m)
        t = pl.read_parquet(RAW_DIR / f"{m}.parquet", columns=["tick", "team_num", "team_rounds_total"])
        last = t.filter(pl.col("tick") == t["tick"].max())
        score = last.drop_nulls("team_num").group_by("team_num").agg(pl.col("team_rounds_total").max())
        s = sorted(score["team_rounds_total"].to_list(), reverse=True) + [0, 0]
        rows.append(dict(match=m, minutes=t["tick"].n_unique() / 64 / 60,
                         rounds=len(ev.get("round_officially_ended", [])),
                         score_diff=s[0] - s[1]))
    return pl.DataFrame(rows)


def team_results(matches) -> pl.DataFrame:
    """Her oyuncu için maç sonucu: kazanan / kaybeden / berabere.

    Devre arasında taraflar (team_num) değişir ve skor oyuncularla birlikte yer değiştirir. Bu yüzden
    oyuncunun takımı SON tick'teki tarafıdır (en sık bulunduğu taraf DEĞİL: 13-2 gibi kısa maçlarda
    oyuncu ilk tarafta daha uzun kalır, en sık taraf yanlış takımı verir).
    """
    rows = []
    for m in matches:
        t = pl.read_parquet(RAW_DIR / f"{m}.parquet", columns=["tick", "steamid", "team_num", "team_rounds_total"])
        last = t.filter(pl.col("tick") == t["tick"].max()).drop_nulls("team_num")
        sc = last.group_by("team_num").agg(pl.col("team_rounds_total").max()).sort("team_rounds_total", descending=True)
        win = sc["team_num"][0] if sc.height == 2 and sc["team_rounds_total"][0] > sc["team_rounds_total"][1] else None
        team = t.drop_nulls("team_num").sort("tick").group_by("steamid").agg(pl.col("team_num").last())
        for sid, tn in team.iter_rows():
            res = "berabere" if win is None else ("kazanan" if tn == win else "kaybeden")
            rows.append(dict(player=f"{m}_{sid}", result=res))
    return pl.DataFrame(rows)


def run_pairs():
    """Harita + mod + uzunluk eşleşmeli çiftlerde (data/pairs.csv) ana deneyler. Bölme birimi = çift."""
    pairs = pl.read_csv("data/pairs.csv")
    pair_of = pl.concat([pairs.select("pair", match="cheat"), pairs.select("pair", match="clean")])
    catalog = pl.read_csv("data/catalog.csv").select("match", "size_mb", "server", "avg_rank")
    stats = match_stats(pair_of["match"].to_list())
    meta = pair_of.join(catalog, on="match").join(stats, on="match").with_columns(
        y=pl.col("match").str.starts_with("with_").cast(pl.Int8))

    print("== Eşleştirme kontrolü (maç düzeyi medyan) ==")
    print(meta.group_by("y").agg(pl.len().alias("maç"), *[pl.col(c).median().round(1)
          for c in ["size_mb", "minutes", "rounds", "score_diff"]]).sort("y").to_pandas().to_string(index=False))

    # join satır sırasını korumaz → sabit sıra (aksi halde eşit skorlarda AUC 3. ondalıkta oynuyor)
    df = (load().join(pair_of, on="match").join(meta.drop("y", "pair"), on="match")
            .sort("match", "player", "tick"))
    # Bir çiftin bir maçı analizden düştüyse çifti tamamen çıkar (denge bozulmasın)
    complete = df.group_by("pair").agg(pl.col("y").n_unique().alias("k")).filter(pl.col("k") == 2)["pair"]
    df = df.filter(pl.col("pair").is_in(complete.to_list()))
    print(f"\nAnalize giren: {df['pair'].n_unique()} çift, {df['match'].n_unique()} maç, "
          f"{df.height} öldürme, {df['player'].n_unique()} oyuncu\n")
    ev = lambda d, f: evaluate(d, f, group_col="pair")  # noqa: E731

    print("== Davranış dışı kontroller (eşleştirme işe yaradıysa ~0.5 beklenir) ==")
    for name, feats in [("dosya boyutu", ["size_mb"]), ("süre (dk)", ["minutes"]), ("round sayısı", ["rounds"]),
                        ("skor farkı (tek taraflılık)", ["score_diff"])]:
        print(f"{name:28s} {fmt(ev(df, feats))}")
    for name, cols in [("server", ["server"]), ("avg_rank", ["avg_rank"])]:
        d, feats = one_hot(df, cols)
        print(f"{name:28s} {fmt(ev(d, feats))}")

    print("\n== Özellik ablasyonu ==")
    for name, feats in FEATURE_SETS.items():
        r = ev(df, feats)
        print(f"{name:28s} {fmt(r)} | {r['n_players']} oy.")
    no_shots = [f for f in AIM_FEATURES if f != "shots"]
    print(f"{'nişan (shots hariç)':28s} {fmt(ev(df, no_shots))}")

    print("\n== Silah sınıfı içinde, sadece nişan ==")
    for wc in ["sniper", "tabanca", "otomatik"]:
        sub = df.filter(pl.col("weapon_class") == wc)
        r = ev(sub, AIM_FEATURES)
        print(f"{wc:10s} {fmt(r)} | {r['n_kills']} öld., {r['n_players']} oy., {r['n_matches']} maç")

    print("\n== Grup içi: temiz oyuncu skoru tek taraflı/kısa maçlarda yükseliyor mu? ==")
    from scipy.stats import spearmanr
    r = ev(df, MAIN)
    p = aggregate(df, r["oof"], group_col="pair").join(meta.select("match", "minutes", "score_diff"), on="match")
    for g, name in [(0, "temiz"), (1, "hileci")]:
        q = p.filter(pl.col("y") == g)
        for c in ["minutes", "score_diff"]:
            rho, pv = spearmanr(q["score"], q[c])
            print(f"{name:7s} skor ~ {c:10s} rho={rho:+.2f} (p={pv:.3f}, n={q.height})")

    print("\n== Kazanan / kaybeden takım (oyuncu takımı = son tick'teki taraf) ==")
    p = p.join(team_results(df["match"].unique().to_list()), on="player", how="left")
    print(p.group_by("y", "result").agg(pl.len().alias("oyuncu"), pl.col("score").mean().round(3))
           .sort("y", "result").to_pandas().to_string(index=False))
    ch = p.filter(pl.col("y") == 1)
    for res in ["kazanan", "kaybeden"]:
        cl = p.filter((pl.col("y") == 0) & (pl.col("result") == res))
        q = pl.concat([ch, cl])
        print(f"hileci vs temiz-{res:8s} oyuncu AUC = {safe_auc(q['y'], q['score']):.3f}  ({ch.height} vs {cl.height})")
    for res in ["kazanan", "kaybeden"]:
        q = p.filter(pl.col("result") == res)
        print(f"{res} hileci vs {res} temiz  oyuncu AUC = {safe_auc(q['y'], q['score']):.3f}  "
              f"({(q['y'] == 1).sum()} vs {(q['y'] == 0).sum()})")


if __name__ == "__main__":
    {"metadata": run_metadata, "maps": run_maps, "aggregation": run_aggregation,
     "matched": run_matched, "pairs": run_pairs}[sys.argv[1]]()
