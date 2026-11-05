"""Harita-eşleşmeli maç seçimi: her haritadan temiz ve hileli gruptan EŞİT sayıda maç.

Önce zaten indirilmiş uygun maçları kullanır, eksikleri sabit seed ile rastgele seçer.
    python src/select_matched.py            -> data/matched_manifest.txt
    python src/select_matched.py --download -> ayrıca eksikleri indirir
"""
import argparse
import random
from pathlib import Path

import polars as pl
from huggingface_hub import hf_hub_download

from download import OUT_DIR, REPO, write_manifest

# Harita -> grup başına maç sayısı (iki grupta da yeterli maç olan haritalar)
PER_MAP = {"de_dust2": 5, "de_mirage": 5, "de_inferno": 5,
           "de_vertigo": 4, "de_nuke": 4, "de_train": 4, "de_anubis": 4, "de_ancient": 4, "cs_office": 4}
FOLDERS = ["no_cheater_present", "with_cheater_present"]
SEED = 42


def select() -> list[str]:
    catalog = pl.read_csv("data/catalog.csv")
    local = {p.with_suffix("").relative_to(OUT_DIR).as_posix() for p in Path(OUT_DIR).glob("*/*.parquet")}
    chosen = []
    for m, k in PER_MAP.items():
        for folder in FOLDERS:
            cands = sorted(catalog.filter((pl.col("map") == m) & (pl.col("folder") == folder))["match"].to_list())
            have = [c for c in cands if c in local]
            rest = [c for c in cands if c not in local]
            rng = random.Random(f"{SEED}-{m}-{folder}")
            chosen += have[:k] + rng.sample(rest, max(0, k - len(have)))
    return sorted(chosen)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--download", action="store_true")
    args = ap.parse_args()

    chosen = select()
    Path("data/matched_manifest.txt").write_text("\n".join(chosen) + "\n", encoding="utf-8")
    catalog = pl.read_csv("data/catalog.csv").filter(pl.col("match").is_in(chosen))
    missing = [m for m in chosen if not (Path(OUT_DIR) / f"{m}.parquet").exists()]
    mb = catalog.filter(pl.col("match").is_in(missing))["size_mb"].sum()
    print(f"{len(chosen)} maç seçildi, {len(missing)} eksik (~{mb / 1e3:.1f} GB)")

    if args.download:
        for i, m in enumerate(missing, 1):
            print(f"[{i}/{len(missing)}] {m}", flush=True)
            for ext in (".parquet", ".json"):
                hf_hub_download(REPO, m + ext, repo_type="dataset", local_dir=OUT_DIR)
        write_manifest()
        print("Bitti.")


if __name__ == "__main__":
    main()
