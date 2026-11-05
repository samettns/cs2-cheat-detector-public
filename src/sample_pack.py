"""Örnek / temiz kurulum için anonim örnek analiz paketi → samples/analysis/<id>/ (git'te; reports/analysis/ git dışı).

Yalnız arayüzün okuduğu dosyalar kopyalanır (ham .dem, kimlik eşlemesi, _stages/ ara dosyaları GİRMEZ).
Kopyalamadan sonra gizlilik denetimi: Steam ID deseni, oyuncu sütunlarında yalnız Player_N, dosya adı/yol yok.

    python src/sample_pack.py                      → varsayılan örnekler (CS2CD örnek seti)
    python src/sample_pack.py <analiz_id> [...]    → ek örnek (ör. Valve; KULLANICI ONAYIYLA)
    python src/sample_pack.py check                → samples/ gizlilik denetimi
CS2CD örnekleri CC-BY-4.0 veri setinden türetilmiştir (atıf: README).
"""
import json
import re
import shutil
import sys
from pathlib import Path

import polars as pl

SRC = Path("reports/analysis")
DST = Path("samples/analysis")
# örnek seti: doğrulanmış pozitif, hilecisiz maç, bilinen yanlış alarm
DEFAULT = ["cs2cd_with_cheater_present_36", "cs2cd_no_cheater_present_146", "cs2cd_no_cheater_present_95"]
FILES = ["metadata.json", "qc.json", "qc.md", "features.parquet", "kill_windows.parquet", "map_context.parquet",
         "utility.parquet", "kills.parquet",
         # Faz 2 (güncel demo) veya OOF (CS2CD eğitim havuzu) çıktıları — hangisi varsa
         "player_scores.parquet", "event_scores.parquet", "map_events.parquet", "model_result.json", "model_report.md",
         "distribution_check.json", "oof_player_scores.parquet", "oof_event_scores.parquet", "oof_map_events.parquet",
         "oof_result.json", "dataset_labels.parquet"]
STEAMID = re.compile(r"7656119\d{10}")
PLAYER_COLS = {"player", "attacker", "victim"}
PLAYER_OK = re.compile(r"^(|Player_\d+|.+_Player_\d+)$")  # boş = dünya/düşme ölümü; CS2CD tam kimlik "<klasör>/<maç>_Player_N"


def privacy_problems(work: Path) -> list[str]:
    out = []
    for f in sorted(work.rglob("*")):
        if f.suffix in (".dem",) or f.name.endswith(".ids.json") or "_stages" in f.parts:
            out.append(f"yasak dosya: {f.name}")
        elif f.suffix in (".json", ".md"):
            t = f.read_text(encoding="utf-8")
            if STEAMID.search(t):
                out.append(f"Steam ID deseni: {f.name}")
            if re.search(r"\.dem\b", t):
                out.append(f".dem dosya adı/yolu: {f.name}")
        elif f.suffix == ".parquet":
            df = pl.read_parquet(f)
            for c, t in df.schema.items():
                if t != pl.Utf8:
                    continue
                vals = df[c].drop_nulls().unique()
                if vals.str.contains(STEAMID.pattern).any():
                    out.append(f"Steam ID deseni: {f.name}:{c}")
                if c in PLAYER_COLS and not vals.str.contains(PLAYER_OK.pattern).all():
                    out.append(f"anonim olmayan oyuncu adı: {f.name}:{c}")
    return out


def pack(ids: list[str]) -> None:
    for did in ids:
        src, dst = SRC / did, DST / did
        if not (src / "metadata.json").exists():
            sys.exit(f"analiz yok: {src}")
        if dst.exists():
            shutil.rmtree(dst)
        dst.mkdir(parents=True)
        for f in FILES:
            if (src / f).exists():
                shutil.copy2(src / f, dst / f)
        probs = privacy_problems(dst)
        if probs:
            shutil.rmtree(dst)
            sys.exit(f"{did}: gizlilik denetimi BAŞARISIZ, kopya silindi: {probs}")
        size = sum(p.stat().st_size for p in dst.iterdir()) / 1e6
        print(f"{did}: {len(list(dst.iterdir()))} dosya, {size:.1f} MB, gizlilik denetimi geçti")


def check() -> int:
    bad = {d.name: privacy_problems(d) for d in sorted(DST.iterdir()) if d.is_dir()} if DST.exists() else {}
    for k, v in bad.items():
        print(k, "OK" if not v else v)
    return int(any(bad.values()))


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[:1] == ["check"]:
        sys.exit(check())
    pack(a or DEFAULT)
    (DST.parent / "README.md").write_text(
        "# Örnek analizler (anonim)\n\nArayüz bu klasördeki analizleri `reports/analysis/` ile birlikte listeler.\n"
        "Ham demo, Steam ID veya gerçek oyuncu adı içermez (`python src/sample_pack.py check`).\n\n"
        "CS2CD örnekleri: CS2CD — Counter-Strike 2 Cheat Detection veri seti "
        "(Hugging Face `CS2CD/CS2CD.Counter-Strike_2_Cheat_Detection`), CC-BY-4.0. Bu klasördeki dosyalar o veriden "
        "türetilmiş özellik ve model çıktılarıdır; skorlar maçı görmemiş kat modellerinden (OOF).\n", encoding="utf-8")
