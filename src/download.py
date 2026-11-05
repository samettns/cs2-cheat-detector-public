"""CS2CD veri setinden belirli sayıda maç indirir.

Kullanım:
    python src/download.py --n 5 --max-mb 60

Her maç iki dosyadır: <id>.parquet (tick verisi) ve <id>.json (olaylar + etiketler).
Seçim sabit bir seed ile rastgele yapılır; aynı komut her zaman aynı maçları indirir.
"""
import argparse
import random
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

REPO = "CS2CD/CS2CD.Counter-Strike_2_Cheat_Detection"
FOLDERS = ["no_cheater_present", "with_cheater_present"]
OUT_DIR = "data/raw"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5, help="her klasörden kaç maç")
    ap.add_argument("--max-mb", type=float, default=None, help="bu boyuttan büyük maçları atla")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    api = HfApi()
    parquets = [
        f for f in api.list_repo_tree(REPO, repo_type="dataset", recursive=True)
        if f.path.endswith(".parquet")
    ]

    for folder in FOLDERS:
        # Her klasörün kendi üreteci: bir klasörde --n değişince diğerinin seçimi değişmesin
        rng = random.Random(f"{args.seed}-{folder}")
        candidates = [f for f in parquets if f.path.startswith(folder + "/")]
        if args.max_mb:
            candidates = [f for f in candidates if f.size / 1e6 <= args.max_mb]
        candidates.sort(key=lambda f: f.path)  # seed'in anlamlı olması için sabit sıra
        chosen = rng.sample(candidates, args.n)

        for f in chosen:
            match_id = f.path.removesuffix(".parquet")
            print(f"İndiriliyor: {match_id}  ({f.size / 1e6:.1f} MB)")
            for ext in (".parquet", ".json"):
                hf_hub_download(REPO, match_id + ext, repo_type="dataset", local_dir=OUT_DIR)

    write_manifest()
    print("Bitti. Dosyalar:", OUT_DIR)


def write_manifest():
    """data/raw içindeki tüm maçları data/manifest.txt'ye yazar (git'e girer, tekrarlanabilirlik için)."""
    matches = sorted(p.with_suffix("").relative_to(OUT_DIR).as_posix()
                     for p in Path(OUT_DIR).glob("*/*.parquet"))
    Path("data/manifest.txt").write_text("\n".join(matches) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
