"""Demo düzeyi dağılım uyarısı. Yöntem özeti: MODEL_CARD.md.

YALNIZ GÖSTERİM: Model B v1, skorlar, G0/G1 eşikleri ve inceleme önceliği bu modülden etkilenmez.
Uyarı demo düzeyindedir (tek tek oyuncular hakkında değil).

İki metrik birlikte değerlendirilir:
  M1 aim_p99      : canlı oyuncularda tick'ten tick'e nişan değişiminin P99'u (qc.json `aim_step_p99`)
  M2 reaction_med : otomatik silah öldürmelerinde medyan reaction_ms (features.parquet; >= 20 dolu öldürme)
Referans: Model v1 eğitimindeki 160 CS2CD temiz maçının maç başı değerleri → %2.5–%97.5 (iç), %0.5–%99.5 (dış).

    python src/distribution_check.py build            → data/reference/v1/distribution_reference.json (+ maç başı parquet)
    python src/distribution_check.py check <analiz>   → reports/analysis/<id>/ için sonucu yazdırır
    python src/distribution_check.py report           → CS2CD + reports/analysis/ özet tablosu (kontrol)
"""
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

REF_DIR = Path("data/reference/v1")
REF_FILE = REF_DIR / "distribution_reference.json"
REF_MATCHES = REF_DIR / "distribution_reference_matches.parquet"
PAIRS = ("data/pairs.csv", "data/pairs_new.csv")
MIN_REACTION_KILLS = 20
INNER = (2.5, 97.5)
OUTER = (0.5, 99.5)

METRICS = {
    "aim_p99": dict(label="Nişan değişimi P99 (tick başına derece)", unit="°",
                    high="daha hızlı/büyük nişan hareketleri", low="daha yavaş nişan hareketleri"),
    "reaction_med": dict(label="Medyan tepki süresi (otomatik silah)", unit=" ms",
                         high="daha uzun tepki süresi", low="daha kısa tepki süresi"),
}
TEXT = {
    2: ("Bu demonun davranış dağılımı Model v1'in temiz eğitim referansından farklı. "
        "Yüksek beceri / dağılım farkı nedeniyle yanlış alarm riski artabilir."),
    1: ("Dağılım farkı gözlendi: bu demonun bir davranış ölçüsü Model v1'in temiz eğitim referansının olağan "
        "aralığının biraz dışında. Skorları yorumlarken dikkate alın."),
    0: "Demo, Model v1'in temiz eğitim referansının olağan aralığında.",
}
TITLE = {2: "Dağılım farkı: temiz eğitim referansından farklı", 1: "Dağılım farkı gözlendi", 0: "Referans aralığında"}
NOTE = ("Bu bilgi demo düzeyindedir, tek tek oyuncular hakkında değildir; skorları ve inceleme önceliğini "
        "değiştirmez. İlgili sınırlılık: professional/high-skill false-positive risk (README → Bilinen sınırlılıklar).")


# ---------- metrikler (demo ve CS2CD için aynı) ----------
def reaction_median(kills: pl.DataFrame) -> tuple[float | None, int]:
    """Otomatik silah öldürmelerinde medyan reaction_ms; dolu öldürme < 20 ise None. (değer, n)"""
    r = kills.filter(pl.col("weapon_class") == "otomatik")["reaction_ms"].drop_nulls().drop_nans()
    return (round(float(r.median()), 1) if r.len() >= MIN_REACTION_KILLS else None), r.len()


# ---------- referans (bir kez, ham CS2CD ile) ----------
def build() -> dict:
    from analyze_demo import aim_step_p99
    from load import RAW_DIR
    from wallhack_eval import dataset

    pairs = pl.concat([pl.read_csv(f) for f in PAIRS])
    clean = pairs.select("map", match="clean").sort("match")
    kills = dataset(pairs_files=PAIRS).filter(pl.col("match").is_in(clean["match"].to_list()))
    rows = []
    for i, (mp, m) in enumerate(clean.iter_rows()):
        rm, n = reaction_median(kills.filter(pl.col("match") == m))
        rows.append(dict(match=m, map=mp, aim_p99=aim_step_p99(RAW_DIR / m), reaction_med=rm, reaction_n=n))
        print(f"{i + 1}/{clean.height} {m}", flush=True)
    per = pl.DataFrame(rows)
    per.write_parquet(REF_MATCHES)

    def ranges(v: np.ndarray) -> dict:
        q = lambda p: round(float(np.percentile(v, p)), 3)   # noqa: E731
        return dict(n=int(len(v)), inner=[q(INNER[0]), q(INNER[1])], outer=[q(OUTER[0]), q(OUTER[1])],
                    median=q(50), min=round(float(v.min()), 3), max=round(float(v.max()), 3))

    ref = dict(
        version="v1", rule="MODEL_CARD.md", decision="",
        universe="CS2CD temiz eğitim maçları: data/pairs.csv + data/pairs_new.csv `clean` (Model v1 havuzu)",
        inner_pct=list(INNER), outer_pct=list(OUTER), min_reaction_kills=MIN_REACTION_KILLS,
        metrics={k: ranges(per[k].drop_nulls().to_numpy()) for k in METRICS},
        by_map_diagnostic={mp: {k: ranges(g[k].drop_nulls().to_numpy()) for k in METRICS}
                           for (mp,), g in per.group_by("map") if g.height >= 30},
    )
    REF_FILE.write_text(json.dumps(ref, ensure_ascii=False, indent=1), encoding="utf-8")
    return ref


def load_reference(path: Path = REF_FILE) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


# ---------- karar ----------
def classify(values: dict, ref: dict) -> dict:
    """values: {"aim_p99": float|None, "reaction_med": float|None} → seviye 0/1/2 + metrik ayrıntıları."""
    details, n_inner, n_outer = [], 0, 0
    for k, meta in METRICS.items():
        r, v = ref["metrics"][k], values.get(k)
        d = dict(metric=k, label=meta["label"], unit=meta["unit"], value=v, inner=r["inner"], outer=r["outer"],
                 ref_median=r["median"], status="hesaplanamadı", direction=None)
        if v is not None:
            lo, hi = r["inner"]
            olo, ohi = r["outer"]
            d["direction"] = "high" if v > hi else "low" if v < lo else None
            if v < olo or v > ohi:
                d["status"], n_outer, n_inner = "dış aralığın dışında", n_outer + 1, n_inner + 1
            elif d["direction"]:
                d["status"], n_inner = "iç aralığın dışında", n_inner + 1
            else:
                d["status"] = "aralık içinde"
            if d["direction"]:
                d["explain"] = meta[d["direction"]]
        details.append(d)
    level = 2 if (n_outer >= 1 or n_inner >= 2) else 1 if n_inner == 1 else 0
    return dict(level=level, title=TITLE[level], text=TEXT[level], note=NOTE, scope="demo",
                affects_scores=False, metrics=details, rule=ref.get("rule"))


def check_analysis(analysis_dir: Path, ref: dict | None = None) -> dict:
    """reports/analysis/<id>/ (qc.json + features.parquet) → uyarı sonucu. Ham demo/CS2CD gerekmez."""
    analysis_dir = Path(analysis_dir)
    ref = ref or load_reference()
    qc = json.loads((analysis_dir / "qc.json").read_text(encoding="utf-8"))
    rm, n = reaction_median(pl.read_parquet(analysis_dir / "features.parquet", columns=["weapon_class", "reaction_ms"]))
    out = classify({"aim_p99": qc.get("aim_step_p99"), "reaction_med": rm}, ref)
    out["reaction_n"] = n
    return out


def fmt(x: float, unit: str, show_unit: bool = True) -> str:
    """Gösterim: derece 1 ondalık, ms tam sayı."""
    return (f"{x:.1f}" if unit == "°" else f"{x:.0f}") + (unit if show_unit else "")


def reasons(res: dict) -> str:
    """Hangi metrik neden uyarı verdi (yalnız aralık dışı olanlar), tek satır."""
    return "; ".join(f"{d['label']}: {fmt(d['value'], d['unit'])} ({d['explain']}; olağan aralık "
                     f"{fmt(d['inner'][0], d['unit'], False)}–{fmt(d['inner'][1], d['unit'])})"
                     for d in res["metrics"] if d.get("direction"))


def markdown(res: dict, heading: str = "##") -> str:
    L = [f"{heading} Demo düzeyi dağılım bilgisi: {res['title']}", "", res["text"], "",
         "| metrik | demo | iç aralık (%2.5–%97.5) | dış aralık (%0.5–%99.5) | durum |", "|---|---|---|---|---|"]
    for d in res["metrics"]:
        u = d["unit"]
        v = "—" if d["value"] is None else fmt(d["value"], u)
        why = f" ({d['explain']})" if d.get("explain") else ""
        L.append(f"| {d['label']} | {v} | {fmt(d['inner'][0], u, False)}–{fmt(d['inner'][1], u)} | "
                 f"{fmt(d['outer'][0], u, False)}–{fmt(d['outer'][1], u)} | "
                 f"{d['status']}{why} |")
    return "\n".join(L + ["", f"_{res['note']}_", ""])


def report() -> None:
    """Kontrol tablosu: CS2CD temiz (örneklem içi) + hileli maçlarda seviye oranları, reports/analysis/ demoları."""
    ref = load_reference()
    per = pl.read_parquet(REF_MATCHES)
    lv = [classify(r, ref)["level"] for r in per.select(*METRICS).to_dicts()]
    print(f"CS2CD temiz eğitim (n={len(lv)}): seviye 1 {lv.count(1)}, seviye 2 {lv.count(2)}")
    print(json.dumps(ref["metrics"], ensure_ascii=False))
    for d in sorted(Path("reports/analysis").iterdir()):
        if (d / "qc.json").exists():
            r = check_analysis(d, ref)
            src = json.loads((d / "qc.json").read_text(encoding="utf-8")).get("source")
            print(d.name, src, "seviye", r["level"],
                  [(m["metric"], m["value"], m["status"], m.get("explain")) for m in r["metrics"]], "n_tepki", r["reaction_n"])


def main(argv=None):
    a = argv if argv is not None else sys.argv[1:]
    if a[:1] == ["build"]:
        print(json.dumps(build(), ensure_ascii=False, indent=1))
    elif a[:1] == ["check"]:
        print(markdown(check_analysis(Path(a[1]))))
    elif a[:1] == ["report"]:
        report()
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
