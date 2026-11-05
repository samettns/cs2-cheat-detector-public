"""CS2CD gösterim modu: eğitim havuzundaki maçlarda in-sample yerine OOF (çapraz doğrulama) skorları
+ model çıktısından AYRI veri seti etiketleri.

    python src/oof_display.py no_cheater_present/146     (Faz 1 --cs2cd çıktısı yoksa önce onu üretir)
    python src/oof_display.py jury                         (data/reference/v1/jury_set.json'daki 3 maç)

Çıktı (reports/analysis/cs2cd_<klasör>_<maç>/, git dışı):
    oof_player_scores.parquet, oof_event_scores.parquet, oof_map_events.parquet  (tanımları; skor kaynağı OOF)
    dataset_labels.parquet   oyuncu başına CS2CD etiketi (model çıktısından ayrı alan)
    oof_result.json          mod, uyarılar, eğitim havuzu bilgisi
Dondurulmuş Faz 2 skorlaması değişmez; eğitim maçlarında Faz 2 (in-sample) çıktısı arayüzde gösterilmez.
OOF skoru yalnız değerlendirmede skorlanan öldürmeler için vardır: hileli maçlardaki "belirsiz" etiketliler
değerlendirme dışıdır → skor verilmez.
"""
import json
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
import analyze_demo as AD          # noqa: E402
import score_demo as S             # noqa: E402
from load import RAW_DIR, cheaters, load_events  # noqa: E402

OOF_FILE = Path("data/reference/v1/oof_all_kills.parquet")
PAIRS = ["data/pairs.csv", "data/pairs_new.csv"]
MODE_NOTE = "Eğitim veri havuzundan — Out-of-Fold / çapraz doğrulama skoru"
LABEL_CHEATER = "Veri seti etiketi: hileci (doğrulanmış)"
LABEL_UNCERTAIN = "Veri seti etiketi: belirsiz / etiketsiz"
LABEL_CLEAN = "CS2CD etiketi: temiz (hilecisiz maç, doğrulanmamış)"
NOT_EVALUATED = "değerlendirme dışı (belirsiz etiket)"


def training_matches() -> set[str]:
    p = pl.concat([pl.read_csv(f) for f in PAIRS])
    return set(p["clean"].to_list()) | set(p["cheat"].to_list())


def dataset_labels(match: str, players: list[str]) -> pl.DataFrame:
    ch = cheaters(load_events(RAW_DIR / match))
    cheat_match = match.startswith("with_cheater_present")
    lab = [LABEL_CHEATER if p in ch else (LABEL_UNCERTAIN if cheat_match else LABEL_CLEAN) for p in players]
    return pl.DataFrame(dict(player=players, dataset_label=lab))


def run(match: str, out_root: Path = AD.OUT_ROOT) -> Path:
    work = out_root / ("cs2cd_" + match.replace("/", "_"))
    if not (work / "features.parquet").exists():
        AD.analyze(cs2cd=match, out_root=out_root)
    fr = S.load_frozen()
    qc = S.check_phase1(work, fr["features"])
    if not qc["model_v1_applicable"]:
        raise S.ScoringError("Faz 1: Model v1 için UYGUN DEĞİL")
    is_train = match in training_matches()
    if not is_train:
        raise S.ScoringError(f"{match} eğitim havuzunda değil: OOF skoru yok (bu maç için Faz 2'yi kullanın)")
    meta = json.loads((work / "metadata.json").read_text(encoding="utf-8"))
    feats = pl.read_parquet(work / "features.parquet")
    kills = pl.read_parquet(work / "kills.parquet")

    oof = pl.read_parquet(OOF_FILE).filter(pl.col("match") == match)
    # Aynı tick'te aynı saldırganın çoklu öldürmesi: özellikler aynı → OOF skoru aynı olmalı; tekilleştir
    dup = oof.group_by("player", "tick").agg(n=pl.col("score").n_unique()).filter(pl.col("n") > 1)
    if dup.height:
        raise S.ScoringError("OOF: aynı (oyuncu, tick) için farklı skorlar")
    oof1 = oof.unique(["player", "tick"], keep="first").select("player", "tick", oof_score="score")
    f = feats.join(oof1, on=["player", "tick"], how="left", maintain_order="left")
    scored = f.filter(pl.col("oof_score").is_not_null())
    # Değerlendirmedeki öldürmelerin hepsi eşleşmeli (satır sayısı kontrolü)
    if scored.height != oof.height:
        raise S.ScoringError(f"OOF eşleşmesi eksik: {scored.height} vs {oof.height}")

    vis = S.visibility_context(work)
    players, events, mev = S.build_tables(scored.drop("oof_score"), scored["oof_score"].to_numpy(), kills, vis,
                                          meta["map"], fr)
    all_players = sorted(set(feats["player"].to_list()) | set(players["player"].to_list()),
                         key=lambda s: int(s.split("_")[1]))
    missing = [p for p in all_players if p not in players["player"].to_list()]
    if missing:   # belirsiz etiketliler: değerlendirme dışı
        n = feats.group_by("player").len()
        extra = pl.DataFrame(dict(player=missing)).join(n.rename({"len": "analyzed_kills"}), on="player", how="left")
        extra = extra.with_columns(review_priority=pl.lit(NOT_EVALUATED))
        players = pl.concat([players, extra], how="diagonal_relaxed").sort("player")
    labels = dataset_labels(match, all_players)
    players.write_parquet(work / "oof_player_scores.parquet")
    events.write_parquet(work / "oof_event_scores.parquet")
    mev.write_parquet(work / "oof_map_events.parquet")
    labels.write_parquet(work / "dataset_labels.parquet")
    (work / "oof_result.json").write_text(json.dumps(dict(
        mode="cs2cd_oof", mode_note=MODE_NOTE, match=match, map=meta["map"], training_pool=is_train,
        disclaimer=S.DISCLAIMER, event_note=S.EVENT_NOTE,
        oof_protocol="havuzlu 160 çift, çift GroupKFold 5x10, tekrar-ortalama (freeze.py ile aynı)",
        players_scored=int(players["player_raw_score"].is_not_null().sum()), players_not_evaluated=missing,
        kills_scored=scored.height, model_identity=S.model_identity()), ensure_ascii=False, indent=1), encoding="utf-8")
    return work


if __name__ == "__main__":
    arg = sys.argv[1]
    targets = ([e["match"] for e in json.loads(Path("data/reference/v1/jury_set.json").read_text(encoding="utf-8"))["examples"]]
               if arg == "jury" else [arg])
    for m in targets:
        print(m, "->", run(m), flush=True)
