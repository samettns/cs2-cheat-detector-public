"""Örnek gösterim seti: önceden belirlenmiş bir kurala göre seçim (kural aşağıda özetlenmiştir).

    python src/jury_set.py      → data/reference/v1/jury_set.json

Kural (özet; ayrıntı):
  aday havuzu: havuzlu 160 çift; tek katlı + .tri'li harita; bilinen sorunlu maçlar (no_cheater_present/254, 246, 68) hariç
  sıra: çiftler pair'e göre sıralanır, numpy.random.default_rng(2026).permutation ile karıştırılır
  (a) doğrulanmış pozitif: hileli maçında >= 1 CS2CD hileci etiketli oyuncunun >= 3 öldürmesi + OOF skoru olan ilk çift
  (b) hilecisiz: haritası (a)'dan farklı, temiz maçında >= 8 oyuncusu >= 3 öldürmeli (OOF) ilk çift
  (c) yanlış alarm: hilecisiz maçta sniper ağırlıklı, >= 3 öldürmeli temiz oyuncunun OOF skoru > G1 sniper eşiği;
      bu maçlar arasından (a), (b) hariç, default_rng(2026).choice ile (yoksa aynı kural G0 ile)
Skor büyüklüğü (a) ve (b) için kriter değildir; (c) yalnız önceden tanımlı hata koşulunu kullanır.
"""
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
import radar as R                                       # noqa: E402
from baseline import MIN_KILLS                          # noqa: E402
from style_threshold import player_styles               # noqa: E402

SEED = 2026
RULE_VERSION = "v1 (2026-10-07)"
D13_EXCLUDED = {"no_cheater_present/254", "no_cheater_present/246", "no_cheater_present/68"}
OUT = Path("data/reference/v1/jury_set.json")


def oof_players() -> pl.DataFrame:
    k = pl.read_parquet("data/reference/v1/oof_all_kills.parquet").with_columns(
        is_sniper=(pl.col("weapon_class") == "sniper").cast(pl.Int8))
    styles = player_styles(k.rename({"player_full": "pid"}).drop("player").rename({"pid": "player"}))
    p = (k.group_by("player_full").agg(pl.col("match").first(), pl.col("player").first(), pl.col("y").first(),
                                       score=pl.col("score").mean(), n=pl.len())
           .join(styles.rename({"player": "player_full"}), on="player_full"))
    return p.filter(pl.col("n") >= MIN_KILLS)


def single_level_supported(map_name: str) -> bool:
    info = R.load_info(map_name)
    return bool(info) and not info["multi_level"] and (Path("data/maps") / f"{map_name}.tri").exists()


def select() -> dict:
    pairs = pl.concat([pl.read_csv("data/pairs.csv"), pl.read_csv("data/pairs_new.csv")]).sort("pair")
    order = np.random.default_rng(SEED).permutation(pairs.height)
    shuffled = [pairs.row(int(i), named=True) for i in order]
    pool = [p for p in shuffled if single_level_supported(p["map"])
            and p["clean"] not in D13_EXCLUDED and p["cheat"] not in D13_EXCLUDED]
    players = oof_players()

    a = next(p for p in pool if players.filter((pl.col("match") == p["cheat"]) & (pl.col("y") == 1)).height >= 1)
    b = next(p for p in pool if p["map"] != a["map"]
             and players.filter((pl.col("match") == p["clean"]) & (pl.col("y") == 0)).height >= 8)

    thr = json.loads(Path("models/v1/thresholds.json").read_text(encoding="utf-8"))
    used = {a["cheat"], b["clean"]}
    pool_clean = {p["clean"]: p for p in pool if p["clean"] not in used}
    c_rule = "G1"
    fa = players.filter((pl.col("y") == 0) & (pl.col("stil") == "sniper ağırlıklı")
                        & (pl.col("score") > thr["G1"]["sniper ağırlıklı"]) & pl.col("match").is_in(list(pool_clean)))
    if fa.height == 0:
        c_rule = "G0 (G1 adayı yok)"
        fa = players.filter((pl.col("y") == 0) & (pl.col("stil") == "sniper ağırlıklı") & (pl.col("score") > thr["G0"])
                            & pl.col("match").is_in(list(pool_clean)))
    cands = sorted(fa["match"].unique().to_list())
    c_match = cands[int(np.random.default_rng(SEED).integers(len(cands)))] if cands else None
    c_pair = pool_clean.get(c_match)
    return dict(
        rule_version=RULE_VERSION, seed=SEED, pool_pairs=len(pool),
        examples=[
            dict(role="dogrulanmis_pozitif", label="Doğrulanmış pozitif etiketli örnek (CS2CD hileli maç)",
                 match=a["cheat"], pair=a["pair"], map=a["map"]),
            dict(role="hilecisiz", label="Hilecisiz veri setinden örnek", match=b["clean"], pair=b["pair"], map=b["map"]),
            dict(role="yanlis_alarm", label="Bilinen başarısızlık / yanlış alarm örneği",
                 note="Bilinen sınırlılık: sniper ağırlıklı oyuncularda yanlış alarm riski.",
                 match=c_match, pair=c_pair["pair"] if c_pair else None, map=c_pair["map"] if c_pair else None,
                 threshold_rule=c_rule, candidate_matches=len(cands),
                 flagged_players=sorted(fa.filter(pl.col("match") == c_match)["player"].to_list()) if c_match else []),
        ])


if __name__ == "__main__":
    res = select()
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1))
